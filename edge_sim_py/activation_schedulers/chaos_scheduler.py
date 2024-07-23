""" Contains the David Cherney's agent activation scheduler with chaos elements."""
# EdgeSimPy components
from edge_sim_py.components import *

# Mesa modules
from mesa.time import BaseScheduler as MesaBaseScheduler

import numpy as np

class ChaosScheduler(MesaBaseScheduler):
    """Class responsible for scheduling the events that take place at each step of the simulation model
    including random deactivation of links."""

    def step(self) -> None:
        """Defines what happens at each step of the simulation model.

        Activation order:
            7 Network Links (some disabled)
            5 Users (application paths, coordiates, access patterms)
            1 Edge Servers
            2 Network Flows
            3 Services
            4 Container Registries
            8 Base Stations
            9 Container Layers
            10 Container Images
            11 Container Registries
            12 Applications
        """
        for agent in NetworkLink.all():
            #############################
            ### Deterministic removal.###
            #############################
            # if self.time==10 and agent == NetworkLink.all()[11]:
            #     agent.remove()
            #####################
            ### Chaos removal.###
            #####################
            if self.time > 0: # No chaos in the first step to allow paths to initialize.
                p_deactivate = 0.02 
                p_reactivate = 0.1 
                if agent.active:
                    if np.random.choice([True,False],p=[p_deactivate,1-p_deactivate]):
                        agent.active = False
                        agent.delay = float('inf')
                        agent.failed = True
                        print(f'Link {agent.id} removed at t={self.time}.')
                else:
                    if np.random.choice([True,False],p=[p_reactivate,1-p_reactivate]):
                        agent.active = True
                        agent.delay = agent.original_delay
                        agent.repaired = True
                        print(f"Link {agent.id} replaced  at t={self.time}")
            agent.step()

        # Set topology with new delay times before re-chosing communication paths.
        for agent in EdgeServer.all():
            agent.step()

        for agent in NetworkFlow.all():
            agent.step()

        for agent in Topology.all():
            agent.step()

        for agent in User.all():
            # At each step, a communication path is re-chosen.
            for application in agent.applications:
                agent.set_communication_path(application)
            agent.step()


        for agent in Service.all():
            agent.step()

        for agent in NetworkSwitch.all():
            agent.step()

        for agent in ContainerRegistry.all():
            agent.step()

        other_agents = (
            BaseStation.all()
            + ContainerLayer.all()
            + ContainerImage.all()
            + Application.all()
        )
        for agent in other_agents:
            agent.step()

        # Advancing simulation
        self.steps += 1
        self.time += 1
