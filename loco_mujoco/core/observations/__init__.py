from .base import Observation, ObservationIndexContainer, ObservationContainer, ObservationType, StatefulObservation
from .goals import *

# register all goals
NoGoal.register()
GoalRandomRootVelocity.register()
GoalForwardRootVelocity.register()
GoalTrajRootVelocity.register()
GoalTrajMimic.register()
GoalTrajMimicv2.register()