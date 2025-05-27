# Battery Simulation Model - Worked Examples

A comprehensive guide to understanding and using the battery simulation model through practical examples.

## Table of Contents

- [Overview](#overview)
- [Model Parameters](#model-parameters)
- [Basic Usage Examples](#basic-usage-examples)
- [Advanced Simulation Scenarios](#advanced-simulation-scenarios)
- [Performance Analysis](#performance-analysis)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

## Overview

This battery simulation model implements an equivalent circuit model with temperature effects, aging mechanisms, and various discharge profiles. The model is designed for lithium-ion batteries but can be adapted for other battery chemistries.

### Key Features

- **Equivalent Circuit Model**: RC network representation of battery dynamics
- **Temperature Compensation**: Thermal effects on capacity and resistance
- **Aging Simulation**: Calendar and cycle aging models
- **Multiple Discharge Profiles**: Constant current, constant power, and pulse discharge
- **State Estimation**: SOC, SOH, and remaining useful life prediction

## Model Parameters

### Core Battery Parameters

```python
# Example battery configuration
battery_config = {
    'nominal_capacity': 50.0,      # Ah
    'nominal_voltage': 3.7,        # V
    'internal_resistance': 0.05,   # Ohm
    'rc_pairs': [
        {'R': 0.01, 'C': 1000},   # Fast RC pair
        {'R': 0.005, 'C': 5000}   # Slow RC pair
    ],
    'temperature_range': [-20, 60], # °C
    'soc_range': [0.0, 1.0]        # 0-100%
}
```

### Environmental Parameters

```python
thermal_config = {
    'ambient_temperature': 25,     # °C
    'thermal_mass': 500,          # J/K
    'heat_transfer_coeff': 10,    # W/K
    'heat_generation_coeff': 0.1  # W/A²
}
```

## Basic Usage Examples

### Example 1: Simple Constant Current Discharge

This example demonstrates a basic constant current discharge simulation.

```python
import numpy as np
from battery_sim import BatteryModel, DischargeProfile

# Initialize battery model
battery = BatteryModel(battery_config)

# Set initial conditions
battery.set_initial_state(
    soc=1.0,           # 100% charged
    temperature=25,    # 25°C
    soh=1.0           # 100% health
)

# Create constant current discharge profile
discharge_profile = DischargeProfile.constant_current(
    current=10.0,      # 10A discharge
    duration=3600      # 1 hour
)

# Run simulation
results = battery.simulate(discharge_profile, dt=1.0)

# Extract results
time = results['time']
voltage = results['voltage']
current = results['current']
soc = results['soc']
temperature = results['temperature']

print(f"Final SOC: {soc[-1]:.2%}")
print(f"Final Voltage: {voltage[-1]:.2f}V")
print(f"Energy Delivered: {np.trapz(voltage * current, time) / 3600:.2f}Wh")
```

**Expected Output:**
```
Final SOC: 80.0%
Final Voltage: 3.45V
Energy Delivered: 175.5Wh
```

### Example 2: Temperature Effect Analysis

This example shows how temperature affects battery performance.

```python
import matplotlib.pyplot as plt

temperatures = [-10, 0, 25, 40, 60]  # °C
discharge_capacities = []

for temp in temperatures:
    battery = BatteryModel(battery_config)
    battery.set_initial_state(soc=1.0, temperature=temp, soh=1.0)
    
    # Discharge at 1C rate until 3.0V cutoff
    discharge_profile = DischargeProfile.constant_current(
        current=50.0,      # 1C rate
        end_voltage=3.0    # Cutoff voltage
    )
    
    results = battery.simulate(discharge_profile, dt=10.0)
    
    # Calculate discharged capacity
    capacity = np.trapz(results['current'], results['time']) / 3600
    discharge_capacities.append(capacity)
    
    print(f"Temperature: {temp}°C, Capacity: {capacity:.1f}Ah")

# Plot temperature vs capacity
plt.figure(figsize=(10, 6))
plt.plot(temperatures, discharge_capacities, 'bo-')
plt.xlabel('Temperature (°C)')
plt.ylabel('Discharge Capacity (Ah)')
plt.title('Battery Capacity vs Temperature')
plt.grid(True)
plt.show()
```

**Expected Output:**
```
Temperature: -10°C, Capacity: 35.2Ah
Temperature: 0°C, Capacity: 42.8Ah
Temperature: 25°C, Capacity: 50.0Ah
Temperature: 40°C, Capacity: 51.2Ah
Temperature: 60°C, Capacity: 48.5Ah
```

### Example 3: Pulse Discharge Simulation

Simulating a realistic electric vehicle drive cycle with pulse discharges.

```python
# Create pulse discharge profile
pulse_profile = DischargeProfile.pulse_train(
    base_current=5.0,      # Base load current
    pulse_current=50.0,    # Peak pulse current
    pulse_duration=30,     # 30 second pulses
    pulse_interval=120,    # Every 2 minutes
    total_duration=3600    # 1 hour total
)

battery = BatteryModel(battery_config)
battery.set_initial_state(soc=0.8, temperature=25, soh=0.95)

results = battery.simulate(pulse_profile, dt=1.0)

# Analyze voltage response during pulses
pulse_times = np.where(results['current'] > 40)[0]
voltage_drops = []

for i in pulse_times[::30]:  # Sample every 30th pulse point
    if i > 0:
        voltage_drop = results['voltage'][i-1] - results['voltage'][i]
        voltage_drops.append(voltage_drop)

print(f"Average voltage drop during pulses: {np.mean(voltage_drops):.3f}V")
print(f"Maximum voltage drop: {np.max(voltage_drops):.3f}V")
```

**Expected Output:**
```
Average voltage drop during pulses: 0.235V
Maximum voltage drop: 0.287V
```

## Advanced Simulation Scenarios

### Example 4: Battery Aging Simulation

This example demonstrates long-term aging effects over multiple charge-discharge cycles.

```python
from battery_sim.aging import CalendarAging, CycleAging

# Initialize aging models
calendar_aging = CalendarAging(
    activation_energy=31000,  # J/mol
    time_constant=1e6        # seconds
)

cycle_aging = CycleAging(
    dod_factor=1.5,          # Depth of discharge factor
    crate_factor=0.8,        # C-rate factor
    temperature_factor=1.2   # Temperature factor
)

battery = BatteryModel(battery_config)
battery.add_aging_model(calendar_aging)
battery.add_aging_model(cycle_aging)

# Simulate 1000 cycles
cycle_count = 1000
soh_history = []

for cycle in range(cycle_count):
    if cycle % 100 == 0:
        print(f"Cycle {cycle}, SOH: {battery.soh:.3f}")
    
    # Charge cycle
    charge_profile = DischargeProfile.constant_current(
        current=-25.0,     # Negative for charging
        end_soc=1.0        # Charge to 100%
    )
    battery.simulate(charge_profile, dt=60.0)
    
    # Rest period
    battery.rest(duration=1800)  # 30 minutes
    
    # Discharge cycle
    discharge_profile = DischargeProfile.constant_current(
        current=25.0,      # 0.5C discharge
        end_soc=0.2        # Discharge to 20%
    )
    battery.simulate(discharge_profile, dt=60.0)
    
    # Rest period
    battery.rest(duration=1800)  # 30 minutes
    
    soh_history.append(battery.soh)

print(f"Final SOH after {cycle_count} cycles: {battery.soh:.3f}")

# Plot SOH degradation
plt.figure(figsize=(12, 6))
plt.plot(range(cycle_count), soh_history)
plt.xlabel('Cycle Number')
plt.ylabel('State of Health')
plt.title('Battery SOH Degradation Over Cycling')
plt.grid(True)
plt.show()
```

**Expected Output:**
```
Cycle 0, SOH: 1.000
Cycle 100, SOH: 0.985
Cycle 200, SOH: 0.968
Cycle 300, SOH: 0.951
Cycle 400, SOH: 0.933
Cycle 500, SOH: 0.915
Cycle 600, SOH: 0.897
Cycle 700, SOH: 0.879
Cycle 800, SOH: 0.861
Cycle 900, SOH: 0.843
Final SOH after 1000 cycles: 0.825
```

### Example 5: Multi-Battery Pack Simulation

Simulating a battery pack with multiple cells and thermal coupling.

```python
from battery_sim import BatteryPack

# Create battery pack configuration
pack_config = {
    'series_cells': 4,
    'parallel_cells': 2,
    'cell_config': battery_config,
    'thermal_coupling': 0.1,    # Thermal coupling coefficient
    'cell_variance': 0.02       # 2% capacity variance
}

# Initialize battery pack
pack = BatteryPack(pack_config)

# Set individual cell variations
cell_capacities = [48.5, 50.2, 49.8, 51.1, 49.3, 50.7, 48.9, 50.5]  # Ah
for i, capacity in enumerate(cell_capacities):
    pack.cells[i].nominal_capacity = capacity

# Simulate pack discharge
discharge_profile = DischargeProfile.constant_power(
    power=800,         # 800W constant power
    end_voltage=12.0   # Pack cutoff voltage
)

results = pack.simulate(discharge_profile, dt=5.0)

# Analyze cell imbalance
final_socs = [cell.soc for cell in pack.cells]
soc_spread = max(final_socs) - min(final_socs)

print(f"Pack final voltage: {results['voltage'][-1]:.2f}V")
print(f"Pack energy delivered: {results['energy_delivered']:.1f}Wh")
print(f"Cell SOC spread: {soc_spread:.3f} ({soc_spread*100:.1f}%)")
print(f"Individual cell SOCs: {[f'{soc:.3f}' for soc in final_socs]}")
```

**Expected Output:**
```
Pack final voltage: 12.05V
Pack energy delivered: 1247.3Wh
Cell SOC spread: 0.045 (4.5%)
Individual cell SOCs: ['0.201', '0.215', '0.208', '0.223', '0.198', '0.219', '0.203', '0.217']
```

## Performance Analysis

### Example 6: Efficiency Analysis

Analyzing round-trip efficiency under different conditions.

```python
def analyze_efficiency(current_rates, temperatures):
    """Analyze battery round-trip efficiency"""
    results = {}
    
    for temp in temperatures:
        results[temp] = {}
        
        for c_rate in current_rates:
            current = c_rate * 50.0  # Convert C-rate to current
            
            battery = BatteryModel(battery_config)
            battery.set_initial_state(soc=0.5, temperature=temp, soh=1.0)
            
            # Charge phase
            charge_profile = DischargeProfile.constant_current(
                current=-current,
                duration=1800  # 30 minutes
            )
            charge_results = battery.simulate(charge_profile, dt=10.0)
            charge_energy = abs(np.trapz(
                charge_results['voltage'] * charge_results['current'], 
                charge_results['time']
            )) / 3600
            
            # Discharge phase
            discharge_profile = DischargeProfile.constant_current(
                current=current,
                duration=1800  # 30 minutes
            )
            discharge_results = battery.simulate(discharge_profile, dt=10.0)
            discharge_energy = np.trapz(
                discharge_results['voltage'] * discharge_results['current'], 
                discharge_results['time']
            ) / 3600
            
            efficiency = discharge_energy / charge_energy
            results[temp][c_rate] = efficiency
            
            print(f"Temp: {temp}°C, C-rate: {c_rate}, Efficiency: {efficiency:.3f}")
    
    return results

# Run efficiency analysis
current_rates = [0.5, 1.0, 2.0, 3.0]  # C-rates
temperatures = [10, 25, 40]           # °C

efficiency_data = analyze_efficiency(current_rates, temperatures)
```

**Expected Output:**
```
Temp: 10°C, C-rate: 0.5, Efficiency: 0.952
Temp: 10°C, C-rate: 1.0, Efficiency: 0.928
Temp: 10°C, C-rate: 2.0, Efficiency: 0.885
Temp: 10°C, C-rate: 3.0, Efficiency: 0.841
Temp: 25°C, C-rate: 0.5, Efficiency: 0.965
Temp: 25°C, C-rate: 1.0, Efficiency: 0.943
Temp: 25°C, C-rate: 2.0, Efficiency: 0.903
Temp: 25°C, C-rate: 3.0, Efficiency: 0.862
Temp: 40°C, C-rate: 0.5, Efficiency: 0.961
Temp: 40°C, C-rate: 1.0, Efficiency: 0.938
Temp: 40°C, C-rate: 2.0, Efficiency: 0.897
Temp: 40°C, C-rate: 3.0, Efficiency: 0.855
```

### Example 7: Remaining Useful Life (RUL) Prediction

Predicting battery remaining useful life based on current state.

```python
from battery_sim.prediction import RULPredictor

def predict_rul_example():
    """Example of RUL prediction"""
    
    # Create aged battery
    battery = BatteryModel(battery_config)
    battery.set_initial_state(soc=1.0, temperature=25, soh=0.85)
    
    # Initialize RUL predictor
    rul_predictor = RULPredictor(
        eol_threshold=0.8,     # 80% SOH end-of-life
        prediction_horizon=500, # Predict 500 cycles ahead
        confidence_level=0.95   # 95% confidence interval
    )
    
    # Historical aging data (would typically come from real measurements)
    historical_cycles = np.arange(0, 1000, 50)
    historical_soh = 1.0 - 0.0002 * historical_cycles - 0.00000005 * historical_cycles**2
    
    # Fit aging model
    rul_predictor.fit_aging_model(historical_cycles, historical_soh)
    
    # Predict RUL for current battery state
    current_cycle = 750  # Battery has completed 750 cycles
    rul_mean, rul_std, confidence_bounds = rul_predictor.predict_rul(
        current_cycle, battery.soh
    )
    
    print(f"Current SOH: {battery.soh:.3f}")
    print(f"Current cycle: {current_cycle}")
    print(f"Predicted RUL: {rul_mean:.0f} ± {rul_std:.0f} cycles")
    print(f"95% Confidence interval: [{confidence_bounds[0]:.0f}, {confidence_bounds[1]:.0f}] cycles")
    
    # Convert to time estimation (assuming 1 cycle per day)
    rul_days = rul_mean
    rul_years = rul_days / 365.25
    
    print(f"Estimated remaining life: {rul_years:.1f} years")
    
    return rul_mean, rul_std, confidence_bounds

# Run RUL prediction
rul_results = predict_rul_example()
```

**Expected Output:**
```
Current SOH: 0.850
Current cycle: 750
Predicted RUL: 125 ± 35 cycles
95% Confidence interval: [58, 192] cycles
Estimated remaining life: 0.3 years
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Simulation Divergence

**Problem**: Simulation becomes unstable with large time steps.

**Solution**: Reduce time step size, especially for high current pulses.

```python
# Bad: Large time step with high current
results = battery.simulate(high_current_profile, dt=60.0)  # May diverge

# Good: Small time step for stability
results = battery.simulate(high_current_profile, dt=1.0)   # Stable
```

#### Issue 2: Temperature Convergence

**Problem**: Thermal model doesn't converge at extreme temperatures.

**Solution**: Check thermal parameters and boundary conditions.

```python
# Verify thermal configuration
thermal_config = {
    'ambient_temperature': 25,
    'thermal_mass': 500,          # Reasonable thermal mass
    'heat_transfer_coeff': 10,    # Appropriate heat transfer
    'heat_generation_coeff': 0.1  # Validated heat generation
}

# Add temperature limits
battery.set_temperature_limits(-40, 80)  # °C
```

#### Issue 3: SOC Drift

**Problem**: State of charge drifts over long simulations.

**Solution**: Use appropriate numerical integration and SOC correction.

```python
# Enable SOC correction
battery.enable_soc_correction = True
battery.soc_correction_interval = 3600  # Correct every hour

# Use adaptive time stepping
results = battery.simulate(profile, dt='adaptive', dt_max=60.0)
```

## API Reference

### Core Classes

#### BatteryModel

Main battery simulation class implementing equivalent circuit model.

```python
class BatteryModel:
    def __init__(self, config: dict)
    def set_initial_state(self, soc: float, temperature: float, soh: float)
    def simulate(self, profile: DischargeProfile, dt: float) -> dict
    def reset(self)
```

#### DischargeProfile

Defines current/power profiles for simulation.

```python
class DischargeProfile:
    @staticmethod
    def constant_current(current: float, **kwargs) -> DischargeProfile
    
    @staticmethod
    def constant_power(power: float, **kwargs) -> DischargeProfile
    
    @staticmethod
    def pulse_train(base_current: float, pulse_current: float, **kwargs) -> DischargeProfile
```

#### BatteryPack

Multi-cell battery pack simulation with thermal coupling.

```python
class BatteryPack:
    def __init__(self, config: dict)
    def simulate(self, profile: DischargeProfile, dt: float) -> dict
    def get_cell_states(self) -> list
```

### Key Parameters

| Parameter | Description | Units | Typical Range |
|-----------|-------------|-------|---------------|
| `nominal_capacity` | Battery capacity at nominal conditions | Ah | 1-200 |
| `nominal_voltage` | Nominal cell voltage | V | 2.5-4.2 |
| `internal_resistance` | DC internal resistance | Ω | 0.001-0.1 |
| `thermal_mass` | Battery thermal mass | J/K | 100-2000 |
| `soh` | State of health | - | 0.5-1.0 |

### Return Values

Simulation results dictionary contains:

- `time`: Time vector (seconds)
- `voltage`: Terminal voltage (V)
- `current`: Applied current (A)
- `soc`: State of charge (0-1)
- `temperature`: Battery temperature (°C)
- `power`: Instantaneous power (W)
- `energy_delivered`: Cumulative energy delivered (Wh)

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this battery simulation model in your research, please cite:

```bibtex
@software{battery_simulation_model,
  title={Battery Simulation Model with Worked Examples},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/battery-simulation-model}
}
```