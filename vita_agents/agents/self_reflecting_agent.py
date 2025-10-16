"""
🪞 Self-Reflecting Agent - Metacognitive Performance Evaluation
============================================================

This agent implements self-reflection capabilities for continuous performance
improvement and quality assurance in healthcare AI systems.
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import structlog
from pydantic import BaseModel, Field
import json
import statistics

from ..core.agent import BaseAgent, AgentMessage, MessageType, Priority, AgentStatus


logger = structlog.get_logger(__name__)


class ReflectionType(str, Enum):
    """Types of self-reflection analysis"""
    PERFORMANCE_REVIEW = "performance_review"
    DECISION_ANALYSIS = "decision_analysis"
    OUTCOME_EVALUATION = "outcome_evaluation"
    ERROR_ANALYSIS = "error_analysis"
    IMPROVEMENT_PLANNING = "improvement_planning"
    QUALITY_ASSESSMENT = "quality_assessment"


class PerformanceMetric(str, Enum):
    """Performance metrics for reflection"""
    ACCURACY = "accuracy"
    RESPONSE_TIME = "response_time"
    CONFIDENCE = "confidence"
    USER_SATISFACTION = "user_satisfaction"
    CLINICAL_OUTCOMES = "clinical_outcomes"
    ERROR_RATE = "error_rate"
    DECISION_QUALITY = "decision_quality"


class ReflectionLevel(str, Enum):
    """Levels of reflection depth"""
    SURFACE = "surface"           # Basic metrics review
    ANALYTICAL = "analytical"     # Pattern analysis
    CRITICAL = "critical"         # Deep reasoning evaluation
    STRATEGIC = "strategic"       # Long-term improvement planning


class DecisionOutcome(BaseModel):
    """Outcome tracking for decision evaluation"""
    decision_id: str
    timestamp: datetime
    decision_type: str
    decision_data: Dict[str, Any]
    
    # Outcome tracking
    predicted_outcome: Optional[str] = None
    actual_outcome: Optional[str] = None
    outcome_timestamp: Optional[datetime] = None
    
    # Quality metrics
    accuracy_score: Optional[float] = None
    confidence_score: Optional[float] = None
    user_feedback: Optional[str] = None
    clinical_impact: Optional[str] = None
    
    # Context
    patient_context: Dict[str, Any] = Field(default_factory=dict)
    clinical_context: Dict[str, Any] = Field(default_factory=dict)


class PerformancePattern(BaseModel):
    """Identified performance pattern"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: str
    description: str
    frequency: int
    
    # Pattern characteristics
    triggers: List[str] = Field(default_factory=list)
    conditions: List[str] = Field(default_factory=list)
    outcomes: List[str] = Field(default_factory=list)
    
    # Impact assessment
    impact_severity: str = "medium"  # low, medium, high, critical
    improvement_potential: float = Field(description="Potential for improvement (0-1)")
    
    # Recommendations
    suggested_actions: List[str] = Field(default_factory=list)
    monitoring_requirements: List[str] = Field(default_factory=list)


class ReflectionInsight(BaseModel):
    """Insight generated from self-reflection"""
    insight_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    insight_type: ReflectionType
    
    # Core insight
    title: str
    description: str
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(description="Confidence in insight (0-1)")
    
    # Impact and priority
    priority_level: Priority = Priority.NORMAL
    potential_impact: str = "medium"  # low, medium, high
    affected_areas: List[str] = Field(default_factory=list)
    
    # Action items
    recommendations: List[str] = Field(default_factory=list)
    required_changes: List[str] = Field(default_factory=list)
    monitoring_metrics: List[str] = Field(default_factory=list)
    
    # Implementation
    implementation_complexity: str = "medium"  # low, medium, high
    estimated_effort: Optional[str] = None
    success_criteria: List[str] = Field(default_factory=list)


class SelfReflectionReport(BaseModel):
    """Comprehensive self-reflection report"""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    reflection_period: Dict[str, datetime]
    
    # Performance summary
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    performance_trends: Dict[str, str] = Field(default_factory=dict)  # improving, declining, stable
    
    # Key findings
    insights: List[ReflectionInsight] = Field(default_factory=list)
    patterns: List[PerformancePattern] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    
    # Quality assessment
    overall_quality_score: float = Field(description="Overall quality score (0-1)")
    quality_dimensions: Dict[str, float] = Field(default_factory=dict)
    
    # Improvement plan
    improvement_priorities: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
    success_metrics: List[str] = Field(default_factory=list)
    
    # Risk assessment
    identified_risks: List[str] = Field(default_factory=list)
    mitigation_strategies: List[str] = Field(default_factory=list)


class SelfReflectingAgent(BaseAgent):
    """
    Self-Reflecting Agent with metacognitive capabilities.
    
    Responsibilities:
    - Monitor own performance across multiple dimensions
    - Analyze decision patterns and outcomes
    - Identify improvement opportunities
    - Generate actionable insights for optimization
    - Track progress on self-improvement initiatives
    - Provide quality assurance through continuous monitoring
    """
    
    def __init__(self, agent_id: str = None):
        super().__init__(
            agent_id=agent_id or "self_reflecting_agent",
            name="Self-Reflecting Agent",
            description="Metacognitive agent for performance monitoring and improvement"
        )
        
        # Performance tracking
        self.decision_history: List[DecisionOutcome] = []
        self.performance_data: Dict[str, List[float]] = {}
        self.reflection_history: List[SelfReflectionReport] = []
        
        # Reflection configuration
        self.reflection_frequency = timedelta(hours=24)  # Daily reflection
        self.min_decisions_for_reflection = 10
        self.performance_targets = {
            PerformanceMetric.ACCURACY: 0.85,
            PerformanceMetric.CONFIDENCE: 0.80,
            PerformanceMetric.RESPONSE_TIME: 5000,  # ms
            PerformanceMetric.ERROR_RATE: 0.05
        }
        
        # Pattern detection thresholds
        self.pattern_min_frequency = 3
        self.significant_change_threshold = 0.1  # 10% change
        
        # Last reflection timestamp
        self.last_reflection: Optional[datetime] = None

    async def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming reflection requests"""
        try:
            if message.type == MessageType.TASK:
                task_type = message.content.get("task_type")
                
                if task_type == "perform_reflection":
                    return await self._perform_reflection(message)
                elif task_type == "analyze_decision":
                    return await self._analyze_decision(message)
                elif task_type == "track_outcome":
                    return await self._track_outcome(message)
                elif task_type == "generate_improvement_plan":
                    return await self._generate_improvement_plan(message)
                elif task_type == "quality_assessment":
                    return await self._quality_assessment(message)
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
            
            return await super().process_message(message)
            
        except Exception as e:
            logger.error("Error in self-reflecting agent", error=str(e))
            return AgentMessage(
                type=MessageType.ERROR,
                sender_id=self.agent_id,
                receiver_id=message.sender_id,
                content={"error": str(e)}
            )

    async def _perform_reflection(self, message: AgentMessage) -> AgentMessage:
        """Perform comprehensive self-reflection analysis"""
        reflection_level = message.content.get("reflection_level", ReflectionLevel.ANALYTICAL)
        time_period = message.content.get("time_period_hours", 24)
        
        # Define reflection period
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=time_period)
        
        # Generate reflection report
        report = await self._generate_reflection_report(start_time, end_time, reflection_level)
        
        # Store reflection
        self.reflection_history.append(report)
        self.last_reflection = end_time
        
        # Update performance tracking
        await self._update_performance_tracking(report)
        
        logger.info(
            "Self-reflection completed",
            report_id=report.report_id,
            insights_count=len(report.insights),
            patterns_count=len(report.patterns),
            quality_score=report.overall_quality_score
        )
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender_id=self.agent_id,
            receiver_id=message.sender_id,
            content={
                "reflection_report": report.dict(),
                "status": "completed"
            }
        )

    async def _generate_reflection_report(
        self, 
        start_time: datetime, 
        end_time: datetime,
        reflection_level: ReflectionLevel
    ) -> SelfReflectionReport:
        """Generate comprehensive reflection report"""
        
        # Filter decisions in time period
        period_decisions = [
            d for d in self.decision_history 
            if start_time <= d.timestamp <= end_time
        ]
        
        # Calculate performance metrics
        performance_metrics = await self._calculate_performance_metrics(period_decisions)
        performance_trends = await self._analyze_performance_trends(period_decisions)
        
        # Generate insights
        insights = await self._generate_insights(period_decisions, reflection_level)
        
        # Identify patterns
        patterns = await self._identify_patterns(period_decisions)
        
        # Assess strengths and weaknesses
        strengths, weaknesses = await self._assess_strengths_weaknesses(performance_metrics, patterns)
        
        # Calculate overall quality
        quality_score, quality_dimensions = await self._calculate_quality_score(performance_metrics, insights)
        
        # Generate improvement plan
        improvement_priorities, action_items = await self._generate_improvement_items(insights, patterns)
        
        # Risk assessment
        risks, mitigations = await self._assess_risks(patterns, weaknesses)
        
        return SelfReflectionReport(
            agent_id=self.agent_id,
            reflection_period={"start": start_time, "end": end_time},
            performance_metrics=performance_metrics,
            performance_trends=performance_trends,
            insights=insights,
            patterns=patterns,
            strengths=strengths,
            weaknesses=weaknesses,
            overall_quality_score=quality_score,
            quality_dimensions=quality_dimensions,
            improvement_priorities=improvement_priorities,
            action_items=action_items,
            identified_risks=risks,
            mitigation_strategies=mitigations
        )

    async def _calculate_performance_metrics(self, decisions: List[DecisionOutcome]) -> Dict[str, float]:
        """Calculate performance metrics for the period"""
        if not decisions:
            return {}
        
        metrics = {}
        
        # Accuracy (where actual outcome is known)
        accuracy_decisions = [d for d in decisions if d.actual_outcome is not None]
        if accuracy_decisions:
            correct_decisions = [
                d for d in accuracy_decisions 
                if d.predicted_outcome == d.actual_outcome
            ]
            metrics[PerformanceMetric.ACCURACY] = len(correct_decisions) / len(accuracy_decisions)
        
        # Average confidence
        confidence_scores = [d.confidence_score for d in decisions if d.confidence_score is not None]
        if confidence_scores:
            metrics[PerformanceMetric.CONFIDENCE] = statistics.mean(confidence_scores)
        
        # Response time (would need to be tracked from task execution)
        metrics[PerformanceMetric.RESPONSE_TIME] = 2500  # Placeholder
        
        # Error rate
        error_decisions = [d for d in decisions if d.accuracy_score is not None and d.accuracy_score < 0.5]
        if decisions:
            metrics[PerformanceMetric.ERROR_RATE] = len(error_decisions) / len(decisions)
        
        return metrics

    async def _analyze_performance_trends(self, decisions: List[DecisionOutcome]) -> Dict[str, str]:
        """Analyze performance trends over time"""
        trends = {}
        
        if len(decisions) < 5:  # Need minimum data for trend analysis
            return trends
        
        # Sort decisions by timestamp
        sorted_decisions = sorted(decisions, key=lambda x: x.timestamp)
        
        # Split into early and late periods
        mid_point = len(sorted_decisions) // 2
        early_period = sorted_decisions[:mid_point]
        late_period = sorted_decisions[mid_point:]
        
        # Compare accuracy trends
        early_accuracy = self._calculate_period_accuracy(early_period)
        late_accuracy = self._calculate_period_accuracy(late_period)
        
        if early_accuracy is not None and late_accuracy is not None:
            if late_accuracy > early_accuracy + self.significant_change_threshold:
                trends[PerformanceMetric.ACCURACY] = "improving"
            elif late_accuracy < early_accuracy - self.significant_change_threshold:
                trends[PerformanceMetric.ACCURACY] = "declining"
            else:
                trends[PerformanceMetric.ACCURACY] = "stable"
        
        # Similar analysis for other metrics...
        trends[PerformanceMetric.CONFIDENCE] = "stable"  # Placeholder
        
        return trends

    def _calculate_period_accuracy(self, decisions: List[DecisionOutcome]) -> Optional[float]:
        """Calculate accuracy for a period of decisions"""
        accuracy_decisions = [d for d in decisions if d.actual_outcome is not None]
        if not accuracy_decisions:
            return None
        
        correct = sum(1 for d in accuracy_decisions if d.predicted_outcome == d.actual_outcome)
        return correct / len(accuracy_decisions)

    async def _generate_insights(self, decisions: List[DecisionOutcome], level: ReflectionLevel) -> List[ReflectionInsight]:
        """Generate actionable insights from decision analysis"""
        insights = []
        
        if not decisions:
            return insights
        
        # Insight 1: Decision accuracy patterns
        accuracy_insight = await self._analyze_accuracy_patterns(decisions)
        if accuracy_insight:
            insights.append(accuracy_insight)
        
        # Insight 2: Confidence calibration
        confidence_insight = await self._analyze_confidence_calibration(decisions)
        if confidence_insight:
            insights.append(confidence_insight)
        
        # Insight 3: Error patterns
        error_insight = await self._analyze_error_patterns(decisions)
        if error_insight:
            insights.append(error_insight)
        
        # Add more insights based on reflection level
        if level in [ReflectionLevel.CRITICAL, ReflectionLevel.STRATEGIC]:
            # Deep analysis insights
            strategic_insights = await self._generate_strategic_insights(decisions)
            insights.extend(strategic_insights)
        
        return insights

    async def _analyze_accuracy_patterns(self, decisions: List[DecisionOutcome]) -> Optional[ReflectionInsight]:
        """Analyze accuracy patterns and generate insights"""
        accuracy_decisions = [d for d in decisions if d.actual_outcome is not None]
        if len(accuracy_decisions) < 5:
            return None
        
        correct_decisions = [d for d in accuracy_decisions if d.predicted_outcome == d.actual_outcome]
        accuracy_rate = len(correct_decisions) / len(accuracy_decisions)
        
        target_accuracy = self.performance_targets.get(PerformanceMetric.ACCURACY, 0.85)
        
        if accuracy_rate < target_accuracy:
            return ReflectionInsight(
                insight_type=ReflectionType.PERFORMANCE_REVIEW,
                title="Below Target Accuracy Performance",
                description=f"Current accuracy rate ({accuracy_rate:.2%}) is below target ({target_accuracy:.2%})",
                evidence=[
                    f"Analyzed {len(accuracy_decisions)} decisions with known outcomes",
                    f"Accuracy rate: {accuracy_rate:.2%}",
                    f"Target accuracy: {target_accuracy:.2%}"
                ],
                confidence=0.9,
                priority_level=Priority.HIGH,
                potential_impact="high",
                affected_areas=["decision_quality", "clinical_outcomes"],
                recommendations=[
                    "Review decision-making algorithms",
                    "Enhance training data quality",
                    "Implement additional validation steps",
                    "Consider ensemble methods for critical decisions"
                ],
                required_changes=[
                    "Algorithm calibration",
                    "Decision threshold adjustment"
                ],
                success_criteria=[
                    f"Achieve {target_accuracy:.0%} accuracy rate",
                    "Maintain consistency across decision types"
                ]
            )
        
        return None

    async def _analyze_confidence_calibration(self, decisions: List[DecisionOutcome]) -> Optional[ReflectionInsight]:
        """Analyze confidence calibration"""
        confidence_decisions = [
            d for d in decisions 
            if d.confidence_score is not None and d.actual_outcome is not None
        ]
        
        if len(confidence_decisions) < 5:
            return None
        
        # Simple calibration analysis
        high_confidence = [d for d in confidence_decisions if d.confidence_score > 0.8]
        low_confidence = [d for d in confidence_decisions if d.confidence_score < 0.6]
        
        if high_confidence:
            high_conf_accuracy = sum(
                1 for d in high_confidence 
                if d.predicted_outcome == d.actual_outcome
            ) / len(high_confidence)
            
            if high_conf_accuracy < 0.9:  # High confidence should be highly accurate
                return ReflectionInsight(
                    insight_type=ReflectionType.DECISION_ANALYSIS,
                    title="Poor Confidence Calibration",
                    description="High confidence decisions are not achieving expected accuracy",
                    evidence=[
                        f"{len(high_confidence)} high confidence decisions analyzed",
                        f"High confidence accuracy: {high_conf_accuracy:.2%}",
                        "Expected accuracy for high confidence: >90%"
                    ],
                    confidence=0.8,
                    priority_level=Priority.NORMAL,
                    recommendations=[
                        "Recalibrate confidence scoring algorithm",
                        "Review decision thresholds",
                        "Implement confidence validation testing"
                    ]
                )
        
        return None

    async def _analyze_error_patterns(self, decisions: List[DecisionOutcome]) -> Optional[ReflectionInsight]:
        """Analyze patterns in errors"""
        error_decisions = [
            d for d in decisions 
            if d.actual_outcome is not None and d.predicted_outcome != d.actual_outcome
        ]
        
        if len(error_decisions) < 3:
            return None
        
        # Analyze error contexts
        error_contexts = []
        for decision in error_decisions:
            if decision.clinical_context:
                error_contexts.extend(decision.clinical_context.get("conditions", []))
        
        # Find common error patterns
        from collections import Counter
        context_counts = Counter(error_contexts)
        common_contexts = [ctx for ctx, count in context_counts.most_common(3) if count >= 2]
        
        if common_contexts:
            return ReflectionInsight(
                insight_type=ReflectionType.ERROR_ANALYSIS,
                title="Recurring Error Patterns Identified",
                description=f"Errors commonly occur in contexts: {', '.join(common_contexts)}",
                evidence=[
                    f"Analyzed {len(error_decisions)} error cases",
                    f"Common error contexts: {context_counts}"
                ],
                confidence=0.7,
                priority_level=Priority.NORMAL,
                affected_areas=["decision_accuracy", "clinical_safety"],
                recommendations=[
                    "Enhance training for identified error-prone contexts",
                    "Implement additional validation for these scenarios",
                    "Consider specialist consultation triggers"
                ]
            )
        
        return None

    async def _generate_strategic_insights(self, decisions: List[DecisionOutcome]) -> List[ReflectionInsight]:
        """Generate strategic-level insights"""
        insights = []
        
        # Strategic insight about decision volume and complexity
        decision_types = {}
        for decision in decisions:
            decision_type = decision.decision_type
            if decision_type not in decision_types:
                decision_types[decision_type] = 0
            decision_types[decision_type] += 1
        
        if decision_types:
            most_common_type = max(decision_types, key=decision_types.get)
            
            insights.append(ReflectionInsight(
                insight_type=ReflectionType.IMPROVEMENT_PLANNING,
                title="Decision Type Distribution Analysis",
                description=f"Most common decision type: {most_common_type} ({decision_types[most_common_type]} cases)",
                evidence=[f"Decision distribution: {decision_types}"],
                confidence=0.9,
                priority_level=Priority.LOW,
                recommendations=[
                    "Consider specialization for most common decision types",
                    "Optimize algorithms for frequent use cases"
                ]
            ))
        
        return insights

    async def _identify_patterns(self, decisions: List[DecisionOutcome]) -> List[PerformancePattern]:
        """Identify performance patterns"""
        patterns = []
        
        # Pattern: Time-based performance variation
        time_pattern = await self._analyze_time_patterns(decisions)
        if time_pattern:
            patterns.append(time_pattern)
        
        # Pattern: Context-based performance variation
        context_pattern = await self._analyze_context_patterns(decisions)
        if context_pattern:
            patterns.append(context_pattern)
        
        return patterns

    async def _analyze_time_patterns(self, decisions: List[DecisionOutcome]) -> Optional[PerformancePattern]:
        """Analyze time-based performance patterns"""
        if len(decisions) < 10:
            return None
        
        # Group by hour of day
        hour_performance = {}
        for decision in decisions:
            hour = decision.timestamp.hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            
            if decision.accuracy_score is not None:
                hour_performance[hour].append(decision.accuracy_score)
        
        # Find significant variations
        hour_averages = {
            hour: statistics.mean(scores) 
            for hour, scores in hour_performance.items() 
            if len(scores) >= 3
        }
        
        if len(hour_averages) >= 3:
            min_performance = min(hour_averages.values())
            max_performance = max(hour_averages.values())
            
            if max_performance - min_performance > 0.2:  # 20% variation
                return PerformancePattern(
                    pattern_type="time_based_variation",
                    description=f"Performance varies by time of day: {min_performance:.2%} to {max_performance:.2%}",
                    frequency=len(hour_averages),
                    triggers=["time_of_day"],
                    improvement_potential=0.7,
                    suggested_actions=[
                        "Investigate system load patterns",
                        "Consider time-based algorithm adjustments",
                        "Monitor for fatigue-related patterns"
                    ]
                )
        
        return None

    async def _analyze_context_patterns(self, decisions: List[DecisionOutcome]) -> Optional[PerformancePattern]:
        """Analyze context-based performance patterns"""
        # Group by clinical context
        context_performance = {}
        
        for decision in decisions:
            if decision.clinical_context and decision.accuracy_score is not None:
                contexts = decision.clinical_context.get("conditions", [])
                for context in contexts:
                    if context not in context_performance:
                        context_performance[context] = []
                    context_performance[context].append(decision.accuracy_score)
        
        # Find contexts with poor performance
        poor_contexts = []
        for context, scores in context_performance.items():
            if len(scores) >= 3:
                avg_score = statistics.mean(scores)
                if avg_score < 0.7:  # Below 70% accuracy
                    poor_contexts.append((context, avg_score))
        
        if poor_contexts:
            worst_context, worst_score = min(poor_contexts, key=lambda x: x[1])
            
            return PerformancePattern(
                pattern_type="context_performance_issue",
                description=f"Poor performance in {worst_context} context: {worst_score:.2%} accuracy",
                frequency=len(context_performance[worst_context]),
                triggers=[worst_context],
                conditions=["specific_clinical_context"],
                impact_severity="high",
                improvement_potential=0.8,
                suggested_actions=[
                    f"Enhance training for {worst_context} scenarios",
                    "Implement context-specific validation",
                    "Consider specialist consultation triggers"
                ]
            )
        
        return None

    async def _assess_strengths_weaknesses(
        self, 
        metrics: Dict[str, float], 
        patterns: List[PerformancePattern]
    ) -> Tuple[List[str], List[str]]:
        """Assess strengths and weaknesses"""
        strengths = []
        weaknesses = []
        
        # Assess metrics against targets
        for metric, value in metrics.items():
            target = self.performance_targets.get(metric)
            if target:
                if metric == PerformanceMetric.ERROR_RATE:  # Lower is better
                    if value < target:
                        strengths.append(f"Low {metric}: {value:.2%} (target: <{target:.2%})")
                    else:
                        weaknesses.append(f"High {metric}: {value:.2%} (target: <{target:.2%})")
                else:  # Higher is better
                    if value >= target:
                        strengths.append(f"Good {metric}: {value:.2%} (target: >{target:.2%})")
                    else:
                        weaknesses.append(f"Poor {metric}: {value:.2%} (target: >{target:.2%})")
        
        # Add pattern-based assessments
        for pattern in patterns:
            if pattern.impact_severity in ["high", "critical"]:
                weaknesses.append(f"Performance issue: {pattern.description}")
            elif pattern.improvement_potential > 0.8:
                weaknesses.append(f"Improvement opportunity: {pattern.description}")
        
        # Default strengths if none identified
        if not strengths:
            strengths.append("Stable performance within expected parameters")
        
        return strengths, weaknesses

    async def _calculate_quality_score(
        self, 
        metrics: Dict[str, float], 
        insights: List[ReflectionInsight]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate overall quality score"""
        dimension_scores = {}
        
        # Performance dimension
        perf_scores = []
        for metric, value in metrics.items():
            target = self.performance_targets.get(metric, 0.8)
            if metric == PerformanceMetric.ERROR_RATE:
                score = max(0, 1 - (value / target))  # Inverted for error rate
            else:
                score = min(1, value / target)
            perf_scores.append(score)
        
        dimension_scores["performance"] = statistics.mean(perf_scores) if perf_scores else 0.7
        
        # Quality dimension based on insights
        critical_issues = len([i for i in insights if i.priority_level == Priority.CRITICAL])
        high_issues = len([i for i in insights if i.priority_level == Priority.HIGH])
        
        quality_penalty = (critical_issues * 0.3) + (high_issues * 0.1)
        dimension_scores["quality"] = max(0.3, 1.0 - quality_penalty)
        
        # Reliability dimension (simplified)
        dimension_scores["reliability"] = 0.8  # Placeholder
        
        # Overall score
        overall_score = statistics.mean(dimension_scores.values())
        
        return overall_score, dimension_scores

    async def _generate_improvement_items(
        self, 
        insights: List[ReflectionInsight], 
        patterns: List[PerformancePattern]
    ) -> Tuple[List[str], List[str]]:
        """Generate improvement priorities and action items"""
        priorities = []
        actions = []
        
        # Extract from insights
        for insight in insights:
            if insight.priority_level in [Priority.CRITICAL, Priority.HIGH]:
                priorities.append(insight.title)
                actions.extend(insight.recommendations)
        
        # Extract from patterns
        for pattern in patterns:
            if pattern.improvement_potential > 0.7:
                priorities.append(f"Address {pattern.pattern_type}")
                actions.extend(pattern.suggested_actions)
        
        # Remove duplicates and limit
        priorities = list(set(priorities))[:5]
        actions = list(set(actions))[:10]
        
        return priorities, actions

    async def _assess_risks(
        self, 
        patterns: List[PerformancePattern], 
        weaknesses: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Assess risks and generate mitigation strategies"""
        risks = []
        mitigations = []
        
        # Risk from critical patterns
        for pattern in patterns:
            if pattern.impact_severity == "critical":
                risks.append(f"Critical performance issue: {pattern.description}")
                mitigations.extend(pattern.suggested_actions)
        
        # Risk from significant weaknesses
        for weakness in weaknesses:
            if "poor" in weakness.lower() or "high error" in weakness.lower():
                risks.append(f"Quality risk: {weakness}")
                mitigations.append("Implement additional validation measures")
        
        # Default mitigations
        if not mitigations:
            mitigations.append("Continue regular monitoring and assessment")
        
        return risks, mitigations

    async def _analyze_decision(self, message: AgentMessage) -> AgentMessage:
        """Analyze a specific decision"""
        decision_data = message.data.get("decision_data", {})
        
        # Create decision outcome record
        decision = DecisionOutcome(
            decision_id=decision_data.get("decision_id", str(uuid.uuid4())),
            timestamp=datetime.now(),
            decision_type=decision_data.get("decision_type", "unknown"),
            decision_data=decision_data,
            predicted_outcome=decision_data.get("predicted_outcome"),
            confidence_score=decision_data.get("confidence_score"),
            patient_context=decision_data.get("patient_context", {}),
            clinical_context=decision_data.get("clinical_context", {})
        )
        
        # Store decision
        self.decision_history.append(decision)
        
        # Analyze decision quality
        analysis = await self._analyze_decision_quality(decision)
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"decision_analysis": analysis, "decision_id": decision.decision_id}
        )

    async def _analyze_decision_quality(self, decision: DecisionOutcome) -> Dict[str, Any]:
        """Analyze the quality of a specific decision"""
        analysis = {
            "decision_id": decision.decision_id,
            "timestamp": decision.timestamp.isoformat(),
            "quality_indicators": {}
        }
        
        # Confidence assessment
        if decision.confidence_score is not None:
            if decision.confidence_score > 0.8:
                analysis["quality_indicators"]["confidence"] = "high"
            elif decision.confidence_score > 0.6:
                analysis["quality_indicators"]["confidence"] = "moderate"
            else:
                analysis["quality_indicators"]["confidence"] = "low"
                analysis["recommendations"] = analysis.get("recommendations", [])
                analysis["recommendations"].append("Consider additional validation for low confidence decisions")
        
        # Context completeness
        context_score = 0
        if decision.patient_context:
            context_score += 0.5
        if decision.clinical_context:
            context_score += 0.5
        
        analysis["quality_indicators"]["context_completeness"] = context_score
        
        return analysis

    async def _track_outcome(self, message: AgentMessage) -> AgentMessage:
        """Track the outcome of a previous decision"""
        decision_id = message.data.get("decision_id")
        actual_outcome = message.data.get("actual_outcome")
        outcome_timestamp = message.data.get("outcome_timestamp")
        
        # Find and update decision
        for decision in self.decision_history:
            if decision.decision_id == decision_id:
                decision.actual_outcome = actual_outcome
                decision.outcome_timestamp = datetime.fromisoformat(outcome_timestamp) if outcome_timestamp else datetime.now()
                
                # Calculate accuracy score
                if decision.predicted_outcome == actual_outcome:
                    decision.accuracy_score = 1.0
                else:
                    decision.accuracy_score = 0.0
                
                break
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"status": "outcome_tracked", "decision_id": decision_id}
        )

    async def _generate_improvement_plan(self, message: AgentMessage) -> AgentMessage:
        """Generate improvement plan based on reflection analysis"""
        # Use latest reflection report or generate new one
        if self.reflection_history:
            latest_report = self.reflection_history[-1]
        else:
            # Generate quick reflection
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            latest_report = await self._generate_reflection_report(
                start_time, end_time, ReflectionLevel.ANALYTICAL
            )
        
        # Generate improvement plan
        improvement_plan = {
            "priorities": latest_report.improvement_priorities,
            "action_items": latest_report.action_items,
            "success_metrics": latest_report.success_metrics,
            "timeline": "30_days",
            "review_frequency": "weekly"
        }
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"improvement_plan": improvement_plan}
        )

    async def _quality_assessment(self, message: AgentMessage) -> AgentMessage:
        """Perform quality assessment"""
        assessment_period = message.data.get("assessment_period_hours", 24)
        
        # Get recent decisions
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=assessment_period)
        
        recent_decisions = [
            d for d in self.decision_history 
            if start_time <= d.timestamp <= end_time
        ]
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_performance_metrics(recent_decisions)
        overall_quality, quality_dimensions = await self._calculate_quality_score(quality_metrics, [])
        
        assessment = {
            "assessment_period": {"start": start_time.isoformat(), "end": end_time.isoformat()},
            "decisions_analyzed": len(recent_decisions),
            "quality_metrics": quality_metrics,
            "overall_quality_score": overall_quality,
            "quality_dimensions": quality_dimensions,
            "quality_level": "excellent" if overall_quality > 0.9 else 
                           "good" if overall_quality > 0.8 else
                           "acceptable" if overall_quality > 0.7 else "needs_improvement"
        }
        
        return AgentMessage(
            type=MessageType.RESPONSE,
            sender=self.agent_id,
            receiver=message.sender,
            data={"quality_assessment": assessment}
        )

    async def _update_performance_tracking(self, report: SelfReflectionReport) -> None:
        """Update long-term performance tracking"""
        timestamp = report.timestamp
        
        for metric, value in report.performance_metrics.items():
            if metric not in self.performance_data:
                self.performance_data[metric] = []
            
            self.performance_data[metric].append(value)
            
            # Keep only last 100 data points
            if len(self.performance_data[metric]) > 100:
                self.performance_data[metric] = self.performance_data[metric][-100:]

    async def _on_start(self) -> None:
        """Called when agent starts"""
        logger.info("Self-reflecting agent started", agent_id=self.agent_id)

    async def _on_stop(self) -> None:
        """Called when agent stops"""
        logger.info("Self-reflecting agent stopped", agent_id=self.agent_id)

    def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        return [
            "performance_monitoring",
            "decision_analysis",
            "outcome_tracking",
            "pattern_identification",
            "quality_assessment",
            "improvement_planning",
            "risk_assessment",
            "metacognitive_reflection"
        ]


# Example usage and testing
async def test_self_reflecting_agent():
    """Test the self-reflecting agent"""
    agent = SelfReflectingAgent()
    await agent.start()
    
    # Simulate some decision outcomes
    sample_decisions = [
        {
            "decision_id": "d1",
            "decision_type": "diagnosis",
            "predicted_outcome": "pneumonia",
            "actual_outcome": "pneumonia",
            "confidence_score": 0.85,
            "clinical_context": {"conditions": ["respiratory"]}
        },
        {
            "decision_id": "d2",
            "decision_type": "diagnosis",
            "predicted_outcome": "bronchitis",
            "actual_outcome": "pneumonia",
            "confidence_score": 0.75,
            "clinical_context": {"conditions": ["respiratory"]}
        },
        {
            "decision_id": "d3",
            "decision_type": "treatment",
            "predicted_outcome": "antibiotic_therapy",
            "actual_outcome": "antibiotic_therapy",
            "confidence_score": 0.90,
            "clinical_context": {"conditions": ["infectious"]}
        }
    ]
    
    # Track decisions
    for decision_data in sample_decisions:
        message = AgentMessage(
            type=MessageType.TASK,
            sender="test",
            receiver=agent.agent_id,
            data={"task_type": "analyze_decision", "decision_data": decision_data}
        )
        await agent.process_message(message)
        
        # Track outcome
        outcome_message = AgentMessage(
            type=MessageType.TASK,
            sender="test",
            receiver=agent.agent_id,
            data={
                "task_type": "track_outcome",
                "decision_id": decision_data["decision_id"],
                "actual_outcome": decision_data["actual_outcome"]
            }
        )
        await agent.process_message(outcome_message)
    
    # Perform reflection
    reflection_message = AgentMessage(
        type=MessageType.TASK,
        sender="test",
        receiver=agent.agent_id,
        data={
            "task_type": "perform_reflection",
            "reflection_level": "analytical",
            "time_period_hours": 24
        }
    )
    
    response = await agent.process_message(reflection_message)
    reflection_report = response.data["reflection_report"]
    
    print("Self-Reflection Results:")
    print(f"Overall Quality Score: {reflection_report['overall_quality_score']:.2f}")
    print(f"Insights Generated: {len(reflection_report['insights'])}")
    print(f"Patterns Identified: {len(reflection_report['patterns'])}")
    
    print("\nKey Insights:")
    for insight in reflection_report["insights"]:
        print(f"  - {insight['title']}: {insight['description']}")
    
    print("\nImprovement Priorities:")
    for priority in reflection_report["improvement_priorities"]:
        print(f"  - {priority}")
    
    await agent.stop()


if __name__ == "__main__":
    asyncio.run(test_self_reflecting_agent())