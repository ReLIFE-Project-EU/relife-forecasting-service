# ECM comparison report

- Building: SFH_Greece_1946_1969
- Category: Single Family House
- Country: Greece
- Weather source: pvgis
- Simulations: 7 total, 7 successful, 0 failed
- Total runtime: 167.84 s
- Included scenarios: Baseline (no changes), wall_insulation, window_replacement, roof_insulation, floor_insulation, deep_envelope, deep_retrofit_hp_pv + heat pump COP=3.2
- Generated on: 2026-03-24 17:08
- HTML report: results/ecm_api/sfh_greece_1946_1969__wall_insulation_window_replacement_roof_insulation_floor_insulation_deep_envelope_deep_retrofit_hp_pv__report__pvgis.html

# Process

- The workflow runs a baseline plus the configured renovation scenarios defined in `RENOVATION_SCENARIOS`.
- Each scenario produces hourly ISO 52016 results, annual UNI/TS 11300 primary energy, and scenario-specific CSV exports.
- CO2 is computed with the same core logic used by the API emission endpoints and then aggregated per scenario.
- The presentation focuses on annual scenario deltas so the ranking is easy to read for decision making.

# Generated Data

| Scenario | ISO total [kWh] | Primary energy [kWh] | Final energy [kWh] | CO2 [t/y] | CO2 saving [%] |
| --- | --- | --- | --- | --- | --- |
| Baseline (no changes) | 69,309.68 | 176,233.77 | 160,741.34 | 32.76 | 0.00 |
| wall_insulation | 56,459.28 | 142,720.64 | 129,921.91 | 26.49 | 19.10 |
| window_replacement | 66,425.63 | 168,538.16 | 153,555.31 | 31.30 | 4.40 |
| roof_insulation | 52,395.63 | 133,067.66 | 121,278.72 | 24.72 | 24.50 |
| floor_insulation | 67,829.08 | 172,400.06 | 156,979.01 | 32.00 | 2.30 |
| deep_envelope | 29,402.49 | 72,729.79 | 65,025.60 | 13.31 | 59.40 |

# Energy Results

![](annual_energy.png){width=88%}

- The grouped bars compare annual ISO demand and annual primary energy for each scenario.
- This view highlights whether a retrofit mainly reduces building demand, generation losses, or both.

# CO2 Results

![](annual_co2.png){width=88%}

- Lowest annual CO2: deep_retrofit_hp_pv + heat pump COP=3.2 with 3.52 tCO2eq/year (89.3% vs baseline).
- Lowest annual primary energy: deep_retrofit_hp_pv + heat pump COP=3.2 with 44,611 kWh/year.
- Lowest annual ISO 52016 demand: deep_envelope with 29,402 kWh/year.

# Considerations

- CO2 uses a final-energy carrier split: delivered thermal energy is mapped to natural gas, grid imports to grid electricity, and self-consumed PV to solar PV.
- Country codes outside the explicit IT/DE mappings fall back to EU emission factors so archetypes such as Greece remain covered.
- The comparison is operational and annual: it does not include embodied carbon of retrofit materials or replacement systems.
- Results depend on the selected weather source, scenario definitions, and UNI/TS 11300 configuration used during the run.
