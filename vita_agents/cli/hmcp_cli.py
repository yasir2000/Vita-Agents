#!/usr/bin/env python3
"""
HMCP Agent CLI for Vita Agents

Command-line interface for interacting with HMCP (Healthcare Multi-agent 
Communication Protocol) agents, enabling healthcare professionals and 
systems to communicate with AI agents using standardized healthcare protocols.
"""

import asyncio
import json
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import logging

from vita_agents.agents.hmcp_agent import HMCPAgent
from vita_agents.protocols.hmcp import (
    HMCPMessageType, ClinicalUrgency, HealthcareRole,
    PatientContext, ClinicalContext, hmcp_protocol
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HMCPAgentCLI:
    """Command-line interface for HMCP agents"""
    
    def __init__(self):
        self.agents: Dict[str, HMCPAgent] = {}
        self.current_agent: Optional[str] = None
    
    async def create_agent(self, agent_id: str, config: Dict[str, Any]) -> bool:
        """Create a new HMCP agent"""
        try:
            agent = HMCPAgent(agent_id, config)
            self.agents[agent_id] = agent
            
            if not self.current_agent:
                self.current_agent = agent_id
            
            print(f"✅ Created HMCP agent: {agent_id}")
            print(f"   Role: {config.get('role', 'ai_agent')}")
            print(f"   Capabilities: {', '.join(config.get('capabilities', []))}")
            print(f"   Emergency capable: {config.get('emergency_capable', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create agent {agent_id}: {e}")
            return False
    
    async def list_agents(self):
        """List all available HMCP agents"""
        if not self.agents:
            print("No HMCP agents available.")
            return
        
        print("📋 Available HMCP Agents:")
        print("-" * 50)
        
        for agent_id, agent in self.agents.items():
            status = "🟢 Active" if agent_id == self.current_agent else "⚪ Inactive"
            metrics = agent.get_agent_metrics()
            
            print(f"{status} {agent_id}")
            print(f"   Role: {agent.healthcare_role}")
            print(f"   Messages sent: {metrics['messages_sent']}")
            print(f"   Messages received: {metrics['messages_received']}")
            print(f"   Emergency responses: {metrics['emergency_responses']}")
            print(f"   Avg response time: {metrics['average_response_time']:.2f}s")
            print()
    
    async def switch_agent(self, agent_id: str) -> bool:
        """Switch to a different agent"""
        if agent_id not in self.agents:
            print(f"❌ Agent {agent_id} not found")
            return False
        
        self.current_agent = agent_id
        print(f"✅ Switched to agent: {agent_id}")
        return True
    
    async def send_message(self, receiver_id: str, message_type: str, content: str, 
                          patient_id: Optional[str] = None, urgency: str = "routine"):
        """Send a message to another agent"""
        if not self.current_agent:
            print("❌ No active agent selected")
            return
        
        agent = self.agents[self.current_agent]
        
        try:
            # Parse message type
            msg_type = HMCPMessageType[message_type.upper()]
            clinical_urgency = ClinicalUrgency[urgency.upper()]
            
            # Parse content
            try:
                content_dict = json.loads(content)
            except json.JSONDecodeError:
                content_dict = {"message": content}
            
            success = await agent.send_clinical_message(
                receiver_id=receiver_id,
                message_type=msg_type,
                content=content_dict,
                patient_id=patient_id,
                urgency=clinical_urgency
            )
            
            if success:
                print(f"✅ Message sent to {receiver_id}")
                print(f"   Type: {message_type}")
                print(f"   Urgency: {urgency}")
                if patient_id:
                    print(f"   Patient: {patient_id}")
            else:
                print(f"❌ Failed to send message to {receiver_id}")
                
        except KeyError as e:
            print(f"❌ Invalid parameter: {e}")
        except Exception as e:
            print(f"❌ Error sending message: {e}")
    
    async def initiate_emergency(self, patient_id: str, emergency_type: str, 
                               location: str, details: str = "{}"):
        """Initiate emergency response"""
        if not self.current_agent:
            print("❌ No active agent selected")
            return
        
        agent = self.agents[self.current_agent]
        
        try:
            # Parse details
            try:
                details_dict = json.loads(details)
            except json.JSONDecodeError:
                details_dict = {"description": details}
            
            emergency_id = await agent.initiate_emergency_response(
                patient_id=patient_id,
                emergency_type=emergency_type,
                location=location,
                details=details_dict
            )
            
            print(f"🚨 Emergency response initiated!")
            print(f"   Emergency ID: {emergency_id}")
            print(f"   Patient: {patient_id}")
            print(f"   Type: {emergency_type}")
            print(f"   Location: {location}")
            
        except Exception as e:
            print(f"❌ Error initiating emergency: {e}")
    
    async def coordinate_care(self, patient_id: str, workflow_type: str, 
                            participants: List[str], care_plan: str = "{}"):
        """Coordinate care workflow"""
        if not self.current_agent:
            print("❌ No active agent selected")
            return
        
        agent = self.agents[self.current_agent]
        
        try:
            # Parse care plan
            try:
                care_plan_dict = json.loads(care_plan)
            except json.JSONDecodeError:
                care_plan_dict = {"workflow_type": workflow_type}
            
            workflow_id = await agent.coordinate_care_workflow(
                patient_id=patient_id,
                workflow_type=workflow_type,
                participants=participants,
                care_plan=care_plan_dict
            )
            
            print(f"🤝 Care coordination initiated!")
            print(f"   Workflow ID: {workflow_id}")
            print(f"   Patient: {patient_id}")
            print(f"   Type: {workflow_type}")
            print(f"   Participants: {', '.join(participants)}")
            
        except Exception as e:
            print(f"❌ Error coordinating care: {e}")
    
    async def show_metrics(self, agent_id: Optional[str] = None):
        """Show agent metrics"""
        target_agent = agent_id or self.current_agent
        
        if not target_agent or target_agent not in self.agents:
            print("❌ No valid agent specified")
            return
        
        agent = self.agents[target_agent]
        metrics = agent.get_agent_metrics()
        health = agent.get_health_status()
        
        print(f"📊 Metrics for agent: {target_agent}")
        print("-" * 40)
        print(f"Messages sent: {metrics['messages_sent']}")
        print(f"Messages received: {metrics['messages_received']}")
        print(f"Emergency responses: {metrics['emergency_responses']}")
        print(f"Workflow completions: {metrics['workflow_completions']}")
        print(f"Average response time: {metrics['average_response_time']:.3f}s")
        print(f"Error count: {metrics['error_count']}")
        print(f"Active conversations: {metrics['active_conversations']}")
        print(f"Uptime: {metrics['uptime']:.1f}s")
        print()
        print("🏥 Health Status:")
        print(f"Status: {health['status']}")
        print(f"HMCP client connected: {health['hmcp_client_connected']}")
        print(f"HMCP server running: {health['hmcp_server_running']}")
        print(f"Message queue size: {health['message_queue_size']}")
        print(f"Error rate: {health['error_rate']:.1%}")
    
    async def show_router_status(self):
        """Show HMCP router status"""
        try:
            router_info = hmcp_protocol.router.get_router_info()
            
            print("🌐 HMCP Router Status:")
            print("-" * 30)
            print(f"Connected agents: {router_info['connected_agents']}")
            print(f"Active workflows: {router_info['active_workflows']}")
            print(f"Messages routed: {router_info['messages_routed']}")
            print(f"Emergency protocols active: {router_info['emergency_protocols_active']}")
            print()
            
            if router_info['registered_agents']:
                print("📝 Registered Agents:")
                for agent_info in router_info['registered_agents']:
                    print(f"   {agent_info['agent_id']} ({agent_info['role']})")
                    print(f"     Capabilities: {', '.join(agent_info['capabilities'])}")
                    print(f"     Emergency capable: {agent_info['emergency_capable']}")
            
        except Exception as e:
            print(f"❌ Error getting router status: {e}")


async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="HMCP Agent CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create agent command
    create_parser = subparsers.add_parser('create', help='Create a new HMCP agent')
    create_parser.add_argument('agent_id', help='Agent identifier')
    create_parser.add_argument('--role', default='ai_agent', 
                              choices=['physician', 'nurse', 'pharmacist', 'ai_agent'],
                              help='Healthcare role')
    create_parser.add_argument('--capabilities', nargs='*', default=[],
                              help='Agent capabilities')
    create_parser.add_argument('--emergency-capable', action='store_true',
                              help='Enable emergency response capability')
    
    # List agents command
    subparsers.add_parser('list', help='List all agents')
    
    # Switch agent command
    switch_parser = subparsers.add_parser('switch', help='Switch active agent')
    switch_parser.add_argument('agent_id', help='Agent to switch to')
    
    # Send message command
    send_parser = subparsers.add_parser('send', help='Send message to another agent')
    send_parser.add_argument('receiver_id', help='Receiver agent ID')
    send_parser.add_argument('message_type', 
                           choices=['request', 'notification', 'emergency', 'coordination', 'event'],
                           help='Message type')
    send_parser.add_argument('content', help='Message content (JSON or text)')
    send_parser.add_argument('--patient-id', help='Patient ID for PHI context')
    send_parser.add_argument('--urgency', default='routine',
                           choices=['routine', 'urgent', 'emergency'],
                           help='Clinical urgency level')
    
    # Emergency command
    emergency_parser = subparsers.add_parser('emergency', help='Initiate emergency response')
    emergency_parser.add_argument('patient_id', help='Patient ID')
    emergency_parser.add_argument('emergency_type', help='Type of emergency')
    emergency_parser.add_argument('location', help='Emergency location')
    emergency_parser.add_argument('--details', default='{}', help='Additional details (JSON)')
    
    # Coordinate care command
    coordinate_parser = subparsers.add_parser('coordinate', help='Coordinate care workflow')
    coordinate_parser.add_argument('patient_id', help='Patient ID')
    coordinate_parser.add_argument('workflow_type', help='Workflow type')
    coordinate_parser.add_argument('participants', nargs='+', help='Participating agents')
    coordinate_parser.add_argument('--care-plan', default='{}', help='Care plan (JSON)')
    
    # Metrics command
    metrics_parser = subparsers.add_parser('metrics', help='Show agent metrics')
    metrics_parser.add_argument('--agent-id', help='Specific agent ID')
    
    # Router status command
    subparsers.add_parser('router', help='Show router status')
    
    # Interactive mode command
    subparsers.add_parser('interactive', help='Start interactive mode')
    
    args = parser.parse_args()
    
    cli = HMCPAgentCLI()
    
    if args.command == 'create':
        config = {
            'role': args.role,
            'capabilities': args.capabilities,
            'emergency_capable': args.emergency_capable
        }
        await cli.create_agent(args.agent_id, config)
        
    elif args.command == 'list':
        await cli.list_agents()
        
    elif args.command == 'switch':
        await cli.switch_agent(args.agent_id)
        
    elif args.command == 'send':
        await cli.send_message(
            args.receiver_id, args.message_type, args.content,
            args.patient_id, args.urgency
        )
        
    elif args.command == 'emergency':
        await cli.initiate_emergency(
            args.patient_id, args.emergency_type, args.location, args.details
        )
        
    elif args.command == 'coordinate':
        await cli.coordinate_care(
            args.patient_id, args.workflow_type, args.participants, args.care_plan
        )
        
    elif args.command == 'metrics':
        await cli.show_metrics(args.agent_id)
        
    elif args.command == 'router':
        await cli.show_router_status()
        
    elif args.command == 'interactive':
        await interactive_mode(cli)
        
    else:
        parser.print_help()


async def interactive_mode(cli: HMCPAgentCLI):
    """Interactive CLI mode"""
    print("🏥 HMCP Agent Interactive Mode")
    print("Type 'help' for available commands or 'exit' to quit")
    print("-" * 50)
    
    while True:
        try:
            current = f"({cli.current_agent})" if cli.current_agent else "(no agent)"
            command = input(f"hmcp{current}> ").strip()
            
            if not command:
                continue
                
            if command.lower() in ['exit', 'quit']:
                print("👋 Goodbye!")
                break
                
            if command.lower() == 'help':
                print_interactive_help()
                continue
            
            # Parse interactive commands
            parts = command.split()
            cmd = parts[0].lower()
            
            if cmd == 'create' and len(parts) >= 2:
                agent_id = parts[1]
                config = {
                    'role': 'ai_agent',
                    'capabilities': ['general_healthcare'],
                    'emergency_capable': False
                }
                await cli.create_agent(agent_id, config)
                
            elif cmd == 'list':
                await cli.list_agents()
                
            elif cmd == 'switch' and len(parts) >= 2:
                await cli.switch_agent(parts[1])
                
            elif cmd == 'metrics':
                agent_id = parts[1] if len(parts) > 1 else None
                await cli.show_metrics(agent_id)
                
            elif cmd == 'router':
                await cli.show_router_status()
                
            elif cmd == 'send' and len(parts) >= 4:
                receiver_id = parts[1]
                message_type = parts[2]
                content = ' '.join(parts[3:])
                await cli.send_message(receiver_id, message_type, content)
                
            elif cmd == 'emergency' and len(parts) >= 4:
                patient_id = parts[1]
                emergency_type = parts[2]
                location = ' '.join(parts[3:])
                await cli.initiate_emergency(patient_id, emergency_type, location)
                
            else:
                print("❌ Invalid command. Type 'help' for available commands.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_interactive_help():
    """Print interactive mode help"""
    print("""
📖 Interactive Commands:
  create <agent_id>                     - Create a new agent
  list                                  - List all agents
  switch <agent_id>                     - Switch active agent
  send <receiver> <type> <content>      - Send message
  emergency <patient> <type> <location> - Initiate emergency
  metrics [agent_id]                    - Show metrics
  router                                - Show router status
  help                                  - Show this help
  exit                                  - Exit interactive mode

📝 Message Types: request, notification, emergency, coordination, event
🚨 Emergency Types: cardiac_arrest, respiratory_failure, stroke_alert, sepsis_alert
""")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)