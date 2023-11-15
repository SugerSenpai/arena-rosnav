from pedsim_waypoint_plugin.pedsim_waypoint_generator import OutputData, PedsimWaypointGenerator, InputData, WaypointPluginName, WaypointPlugin
import pedsim_msgs.msg
import rvo2
import numpy as np

@PedsimWaypointGenerator.register(WaypointPluginName.ORCA)
class Plugin_Orca(WaypointPlugin):
    def __init__(self):
        self.simulator = rvo2.PyRVOSimulator(1/60., 1.5, 5, 1.5, 2, 0.4, 2)  
        self.agent_id_map = {}
        self.diameter = 4.0 

    def callback(self, data) -> OutputData:
        
        for agent in data.agents:
            # If agent not in simulator, add it and process any obstacles
            if agent.id not in self.agent_id_map.keys():
                self.agent_id_map[agent.id] = self.simulator.addAgent((agent.pose.position.x, agent.pose.position.y), 1.5, len(data.agents) - 1, 1.5, 2, 0.4, 0.4, (0,0))
            
            self.simulator.processObstacles()
            self.simulator.setAgentPrefVelocity(self.agent_id_map[agent.id], (agent.destination.x - agent.pose.position.x, agent.destination.y - agent.pose.position.y))
            
            
        # Perform simulation step
        self.simulator.doStep()

        def get_feedback(agent: pedsim_msgs.msg.AgentState) -> pedsim_msgs.msg.AgentFeedback:
            res = pedsim_msgs.msg.AgentFeedback()

            agent_id = self.agent_id_map.get(agent.id)

            force = self.simulator.getAgentVelocity(agent_id)

            res.id = agent.id
            res.force.x, res.force.y = force
            res.force.z = 0
            res.unforce = False

            return res
        
        return [get_feedback(agent) for agent in data.agents]
    
    



