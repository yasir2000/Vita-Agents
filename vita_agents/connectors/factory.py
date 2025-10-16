"""
EHR Connector Factory and Manager for Vita Agents.

This module provides a factory for creating EHR connectors and managing
multiple EHR connections simultaneously. It includes connection pooling,
health monitoring, and intelligent routing between different EHR systems.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .base import (
    BaseEHRConnector,
    EHRConnectionConfig,
    EHRVendor,
    SyncResult,
    EHRConnectorError,
    EHRConnectionError,
)
from .epic import EpicConnector
from .cerner import CernerConnector
from .allscripts import AllscriptsConnector

logger = logging.getLogger(__name__)


@dataclass
class EHRConnectionPool:
    """Connection pool for EHR connectors."""
    max_connections: int = 10
    min_connections: int = 2
    connection_timeout: int = 300  # 5 minutes
    idle_timeout: int = 600  # 10 minutes
    
    # Internal state
    active_connections: Dict[str, BaseEHRConnector] = field(default_factory=dict)
    idle_connections: Dict[str, BaseEHRConnector] = field(default_factory=dict)
    connection_timestamps: Dict[str, datetime] = field(default_factory=dict)
    health_status: Dict[str, bool] = field(default_factory=dict)


@dataclass
class EHRSystemStatus:
    """Status information for an EHR system."""
    vendor: EHRVendor
    connection_id: str
    is_connected: bool
    is_healthy: bool
    last_health_check: datetime
    response_time: Optional[float]
    error_count: int = 0
    last_error: Optional[str] = None
    uptime_percentage: float = 100.0


class EHRConnectorFactory:
    """
    Factory for creating and managing EHR connectors.
    
    Provides centralized management of EHR connections with:
    - Connection pooling and reuse
    - Health monitoring and circuit breakers
    - Load balancing across multiple instances
    - Automatic failover and recovery
    """
    
    def __init__(self):
        """Initialize the EHR connector factory."""
        self._connector_classes = {
            EHRVendor.EPIC: EpicConnector,
            EHRVendor.CERNER: CernerConnector,
            EHRVendor.ALLSCRIPTS: AllscriptsConnector,
        }
        
        self._connection_pools: Dict[EHRVendor, EHRConnectionPool] = {}
        self._configurations: Dict[str, EHRConnectionConfig] = {}
        self._system_status: Dict[str, EHRSystemStatus] = {}
        
        # Health monitoring
        self._health_check_interval = 300  # 5 minutes
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def register_connector_class(
        self,
        vendor: EHRVendor,
        connector_class: type
    ) -> None:
        """Register a custom connector class for a vendor."""
        self._connector_classes[vendor] = connector_class
        logger.info(f"Registered custom connector for {vendor}")
    
    def add_configuration(self, config_id: str, config: EHRConnectionConfig) -> None:
        """Add an EHR configuration."""
        self._configurations[config_id] = config
        
        # Initialize connection pool for this vendor if not exists
        if config.vendor not in self._connection_pools:
            self._connection_pools[config.vendor] = EHRConnectionPool()
        
        # Initialize status tracking
        self._system_status[config_id] = EHRSystemStatus(
            vendor=config.vendor,
            connection_id=config_id,
            is_connected=False,
            is_healthy=False,
            last_health_check=datetime.utcnow(),
            response_time=None
        )
        
        logger.info(f"Added EHR configuration: {config_id} ({config.vendor})")
    
    def remove_configuration(self, config_id: str) -> None:
        """Remove an EHR configuration."""
        if config_id in self._configurations:
            del self._configurations[config_id]
        
        if config_id in self._system_status:
            del self._system_status[config_id]
        
        logger.info(f"Removed EHR configuration: {config_id}")
    
    def list_configurations(self) -> Dict[str, EHRConnectionConfig]:
        """List all EHR configurations."""
        return self._configurations.copy()
    
    async def create_connector(
        self,
        config_id: str,
        use_pool: bool = True
    ) -> BaseEHRConnector:
        """
        Create or retrieve an EHR connector.
        
        Args:
            config_id: Configuration identifier
            use_pool: Whether to use connection pooling
            
        Returns:
            EHR connector instance
        """
        if config_id not in self._configurations:
            raise EHRConnectorError(
                f"Configuration not found: {config_id}",
                EHRVendor.EPIC  # Default vendor for error
            )
        
        config = self._configurations[config_id]
        vendor = config.vendor
        
        if use_pool:
            # Try to get from pool first
            connector = await self._get_from_pool(config_id, vendor)
            if connector:
                return connector
        
        # Create new connector
        if vendor not in self._connector_classes:
            raise EHRConnectorError(
                f"No connector class registered for vendor: {vendor}",
                vendor
            )
        
        connector_class = self._connector_classes[vendor]
        connector = connector_class(config)
        
        # Connect the connector
        try:
            await connector.connect()
            
            # Update status
            self._system_status[config_id].is_connected = True
            self._system_status[config_id].is_healthy = True
            
            # Add to pool if enabled
            if use_pool:
                await self._add_to_pool(config_id, vendor, connector)
            
            logger.info(f"Created connector for {config_id} ({vendor})")
            return connector
            
        except Exception as e:
            # Update status on failure
            self._system_status[config_id].is_connected = False
            self._system_status[config_id].is_healthy = False
            self._system_status[config_id].last_error = str(e)
            self._system_status[config_id].error_count += 1
            
            logger.error(f"Failed to create connector for {config_id}: {e}")
            raise
    
    async def _get_from_pool(
        self,
        config_id: str,
        vendor: EHRVendor
    ) -> Optional[BaseEHRConnector]:
        """Get a connector from the connection pool."""
        if vendor not in self._connection_pools:
            return None
        
        pool = self._connection_pools[vendor]
        
        # Check idle connections first
        for conn_id, connector in list(pool.idle_connections.items()):
            if conn_id.startswith(config_id) and connector.is_connected:
                # Move from idle to active
                pool.active_connections[conn_id] = pool.idle_connections.pop(conn_id)
                logger.debug(f"Retrieved connector from idle pool: {conn_id}")
                return connector
        
        # Check if we can reuse an active connection (if not at max capacity)
        if len(pool.active_connections) < pool.max_connections:
            # Look for reusable active connections
            for conn_id, connector in pool.active_connections.items():
                if conn_id.startswith(config_id) and connector.is_connected:
                    logger.debug(f"Reusing active connector: {conn_id}")
                    return connector
        
        return None
    
    async def _add_to_pool(
        self,
        config_id: str,
        vendor: EHRVendor,
        connector: BaseEHRConnector
    ) -> None:
        """Add a connector to the connection pool."""
        if vendor not in self._connection_pools:
            self._connection_pools[vendor] = EHRConnectionPool()
        
        pool = self._connection_pools[vendor]
        
        # Generate unique connection ID
        conn_id = f"{config_id}_{datetime.utcnow().timestamp()}"
        
        # Add to active connections
        pool.active_connections[conn_id] = connector
        pool.connection_timestamps[conn_id] = datetime.utcnow()
        pool.health_status[conn_id] = True
        
        logger.debug(f"Added connector to pool: {conn_id}")
    
    async def return_to_pool(
        self,
        config_id: str,
        connector: BaseEHRConnector
    ) -> None:
        """Return a connector to the pool (move from active to idle)."""
        vendor = connector.vendor
        
        if vendor not in self._connection_pools:
            return
        
        pool = self._connection_pools[vendor]
        
        # Find the connector in active connections
        conn_id = None
        for cid, conn in pool.active_connections.items():
            if conn is connector:
                conn_id = cid
                break
        
        if conn_id:
            # Move to idle if still connected
            if connector.is_connected:
                pool.idle_connections[conn_id] = pool.active_connections.pop(conn_id)
                logger.debug(f"Moved connector to idle pool: {conn_id}")
            else:
                # Remove disconnected connector
                pool.active_connections.pop(conn_id)
                if conn_id in pool.connection_timestamps:
                    del pool.connection_timestamps[conn_id]
                if conn_id in pool.health_status:
                    del pool.health_status[conn_id]
                logger.debug(f"Removed disconnected connector: {conn_id}")
    
    @asynccontextmanager
    async def get_connector(self, config_id: str, use_pool: bool = True):
        """
        Context manager for getting and automatically returning connectors.
        
        Usage:
            async with factory.get_connector("epic_config") as connector:
                result = await connector.get_patient("12345")
        """
        connector = await self.create_connector(config_id, use_pool)
        try:
            yield connector
        finally:
            if use_pool:
                await self.return_to_pool(config_id, connector)
            else:
                await connector.disconnect()
    
    async def get_multiple_connectors(
        self,
        config_ids: List[str],
        use_pool: bool = True
    ) -> Dict[str, BaseEHRConnector]:
        """Get multiple connectors simultaneously."""
        connectors = {}
        
        # Create connectors concurrently
        tasks = {
            config_id: asyncio.create_task(self.create_connector(config_id, use_pool))
            for config_id in config_ids
        }
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        # Process results
        for config_id, result in zip(config_ids, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to create connector for {config_id}: {result}")
            else:
                connectors[config_id] = result
        
        return connectors
    
    async def sync_patient_across_systems(
        self,
        patient_identifier: str,
        config_ids: List[str],
        identifier_type: str = "MRN"
    ) -> Dict[str, SyncResult]:
        """
        Synchronize patient data across multiple EHR systems.
        
        Args:
            patient_identifier: Patient identifier value
            config_ids: List of EHR configuration IDs
            identifier_type: Type of identifier (MRN, SSN, etc.)
            
        Returns:
            Dictionary of sync results per system
        """
        sync_results = {}
        
        # Get connectors for all systems
        connectors = await self.get_multiple_connectors(config_ids)
        
        # Sync patient data from each system
        sync_tasks = {}
        for config_id, connector in connectors.items():
            try:
                # Search for patient using identifier
                search_params = {identifier_type.lower(): patient_identifier}
                task = asyncio.create_task(
                    connector.sync_patient_data(patient_identifier)
                )
                sync_tasks[config_id] = task
            except Exception as e:
                logger.error(f"Failed to start sync for {config_id}: {e}")
        
        # Wait for all sync operations to complete
        if sync_tasks:
            results = await asyncio.gather(*sync_tasks.values(), return_exceptions=True)
            
            for config_id, result in zip(sync_tasks.keys(), results):
                if isinstance(result, Exception):
                    logger.error(f"Sync failed for {config_id}: {result}")
                    # Create error sync result
                    sync_results[config_id] = SyncResult(
                        vendor=connectors[config_id].vendor,
                        sync_mode=connectors[config_id].config.sync_mode,
                        start_time=datetime.utcnow(),
                        end_time=datetime.utcnow(),
                        resources_processed=0,
                        resources_created=0,
                        resources_updated=0,
                        resources_failed=1,
                        errors=[str(result)]
                    )
                else:
                    sync_results[config_id] = result
        
        # Return connectors to pool
        for config_id, connector in connectors.items():
            await self.return_to_pool(config_id, connector)
        
        return sync_results
    
    async def get_system_status(self, config_id: Optional[str] = None) -> Union[EHRSystemStatus, Dict[str, EHRSystemStatus]]:
        """
        Get status information for EHR systems.
        
        Args:
            config_id: Specific configuration ID, or None for all systems
            
        Returns:
            Status for specific system or all systems
        """
        if config_id:
            return self._system_status.get(config_id)
        else:
            return self._system_status.copy()
    
    async def start_health_monitoring(self) -> None:
        """Start background health monitoring for all EHR systems."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started EHR health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop background health monitoring."""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        
        logger.info("Stopped EHR health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Background loop for health monitoring."""
        while self._running:
            try:
                await self._perform_health_checks()
                await self._cleanup_idle_connections()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retrying
    
    async def _perform_health_checks(self) -> None:
        """Perform health checks on all configured systems."""
        health_tasks = []
        
        for config_id in self._configurations.keys():
            task = asyncio.create_task(self._check_system_health(config_id))
            health_tasks.append(task)
        
        if health_tasks:
            await asyncio.gather(*health_tasks, return_exceptions=True)
    
    async def _check_system_health(self, config_id: str) -> None:
        """Check health of a specific EHR system."""
        try:
            async with self.get_connector(config_id, use_pool=True) as connector:
                start_time = datetime.utcnow()
                health_info = await connector.health_check()
                end_time = datetime.utcnow()
                
                # Update status
                status = self._system_status[config_id]
                status.is_healthy = len(health_info.get("errors", [])) == 0
                status.last_health_check = end_time
                status.response_time = (end_time - start_time).total_seconds()
                
                if status.is_healthy:
                    status.error_count = 0
                    status.last_error = None
                else:
                    status.error_count += 1
                    status.last_error = "; ".join(health_info.get("errors", []))
                
                logger.debug(f"Health check completed for {config_id}: healthy={status.is_healthy}")
                
        except Exception as e:
            # Update status on health check failure
            status = self._system_status[config_id]
            status.is_healthy = False
            status.is_connected = False
            status.last_health_check = datetime.utcnow()
            status.error_count += 1
            status.last_error = str(e)
            
            logger.warning(f"Health check failed for {config_id}: {e}")
    
    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections that have exceeded timeout."""
        current_time = datetime.utcnow()
        
        for vendor, pool in self._connection_pools.items():
            # Check idle connections for timeout
            expired_connections = []
            
            for conn_id, connector in pool.idle_connections.items():
                timestamp = pool.connection_timestamps.get(conn_id)
                if timestamp:
                    idle_time = (current_time - timestamp).total_seconds()
                    if idle_time > pool.idle_timeout:
                        expired_connections.append(conn_id)
            
            # Remove expired connections
            for conn_id in expired_connections:
                connector = pool.idle_connections.pop(conn_id)
                if conn_id in pool.connection_timestamps:
                    del pool.connection_timestamps[conn_id]
                if conn_id in pool.health_status:
                    del pool.health_status[conn_id]
                
                try:
                    await connector.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting expired connector {conn_id}: {e}")
                
                logger.debug(f"Cleaned up expired connection: {conn_id}")
    
    async def shutdown(self) -> None:
        """Shutdown the factory and clean up all connections."""
        await self.stop_health_monitoring()
        
        # Disconnect all connections
        for vendor, pool in self._connection_pools.items():
            all_connections = {**pool.active_connections, **pool.idle_connections}
            
            for conn_id, connector in all_connections.items():
                try:
                    await connector.disconnect()
                except Exception as e:
                    logger.warning(f"Error disconnecting {conn_id}: {e}")
        
        # Clear all pools
        self._connection_pools.clear()
        self._system_status.clear()
        
        logger.info("EHR connector factory shutdown completed")


# Global factory instance
ehr_factory = EHRConnectorFactory()