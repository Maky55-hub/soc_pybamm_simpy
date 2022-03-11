from typing import List, Dict, Union

import simpy
import pandas as pd
import pybamm


class SimpleBattery:

    def __init__(self, capacity, soc=0):
        self.capacity = capacity
        self.soc = soc

    def update(self, energy):
        self.soc += energy
        excess_energy = 0

        if self.soc < 0:
            excess_energy = self.soc
            self.soc = 0
        elif self.soc > self.capacity:
            excess_energy = self.soc - self.capacity
            self.soc = self.capacity

        return excess_energy


class SimpyBattery:

    def __init__(self, env, capacity, soc=0):
        # eventually we probably want to pass more args like the model, parameters
        self.env = env
        self.capacity = capacity
        self.soc = soc

        self.last_update = env.now
        self.last_solution = None

        self.parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Chen2020)
        self.model = pybamm.lithium_ion.DFN()

    def update(self, power):
        # copy/paste: no guarantee to what happens in this function ;)

        #time_passed = self.env.now - self.last_update
        if power > 0:
            experiment = pybamm.Experiment([f"charge at {0.1 * power} W for 15 s"])
        elif power < 0:
            experiment = pybamm.Experiment([f"discharge at {-0.1 * power} W for 15 s"])
        else:
            return 0

        sim = pybamm.Simulation(self.model, parameter_values=self.parameter_values, experiment=experiment)
        sim.solve(starting_solution=self.last_solution)
        #sim.solve(starting_solution=last_solution)
        #last_solution = sim.solution
        self.last_solution = sim.solution
        d = self.last_solution['Discharge capacity [A.h]']
        self.soc = self.soc + d.entries[-1]
        sim.plot(['Current [A]', 'Terminal voltage [V]', 'Discharge capacity [A.h]'])
        
        # TODO: return excess_energy if battery is full or empty?


def simulate(env: simpy.Environment,
             battery: Union[SimpleBattery, SimpyBattery],
             power_delta_list: List[float],
             records: List[Dict]):
    for power_delta in power_delta_list:
        yield env.timeout(1)
        excess_energy = battery.update(power_delta)
        records.append({
            "power_delta": power_delta,
            "excess_energy": excess_energy,
            "soc": battery.soc,
        })


def main():
    # For now let's assume the simple case of one step every second where we first (dis)charge and then implicitly read.
    # Later we can extend this to a more asynchronous charge/discharge/read pattern with different processes if we want
    power_delta_list = [1, -3, 2, 3, 4, -2]
    records = []  # log of some infos for later analysis

    env = simpy.Environment()
    # battery = SimpleBattery(capacity=1)
    battery = SimpyBattery(env, capacity=1)
    env.process(simulate(env, battery, power_delta_list, records))
    env.run()

    result = pd.DataFrame(records)
    with open("result.csv", "w") as f:
        f.write(result.to_csv())
    print(result)


if __name__ == '__main__':
    main()
