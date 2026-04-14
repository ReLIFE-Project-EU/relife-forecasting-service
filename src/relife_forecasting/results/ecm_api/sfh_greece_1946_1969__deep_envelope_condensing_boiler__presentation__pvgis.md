# ECM comparison report

- Building: SFH_Greece_1946_1969
- Category: Single Family House
- Country: Greece
- Weather source: pvgis
- Simulations: 3 total, 3 successful, 0 failed
- Total runtime: 0.00 s
- Included scenarios: Baseline, deep_envelope, condensing_boiler + condensing boiler eta_generation=1.1
- Generated on: 2026-03-24 14:18
- HTML report: src/relife_forecasting/results/ecm_api/sfh_greece_1946_1969__deep_envelope_condensing_boiler__report__pvgis.html

# Process

- The workflow runs a baseline plus the configured renovation scenarios defined in `RENOVATION_SCENARIOS`.
- Each scenario produces hourly ISO 52016 results, annual UNI/TS 11300 primary energy, and scenario-specific CSV exports.
- CO2 is computed with the same core logic used by the API emission endpoints and then aggregated per scenario.
- The presentation focuses on annual scenario deltas so the ranking is easy to read for decision making.

# Generated Data

| Scenario | ISO total [kWh] | Primary energy [kWh] | Final energy [kWh] | CO2 [t/y] | CO2 saving [%] |
| --- | --- | --- | --- | --- | --- |
| Baseline | 69,309.68 | 84,572.06 | 77,229.08 | 15.76 | 0.00 |
| deep_envelope | 29,402.49 | 34,999.80 | 31,107.47 | 6.39 | 59.40 |
| condensing_boiler + condensing boiler eta_generation=1.1 | 69,309.68 | 71,435.31 | 64,913.76 | 13.27 | 15.80 |

# Energy Results

![](annual_energy.png){width=88%}

- The grouped bars compare annual ISO demand and annual primary energy for each scenario.
- This view highlights whether a retrofit mainly reduces building demand, generation losses, or both.

# CO2 Results

![](annual_co2.png){width=88%}

- Lowest annual CO2: deep_envelope with 6.39 tCO2eq/year (59.4% vs baseline).
- Lowest annual primary energy: deep_envelope with 35,000 kWh/year.
- Lowest annual ISO 52016 demand: deep_envelope with 29,402 kWh/year.

# Considerations

- CO2 uses a final-energy carrier split: delivered thermal energy is mapped to natural gas, grid imports to grid electricity, and self-consumed PV to solar PV.
- Country codes outside the explicit IT/DE mappings fall back to EU emission factors so archetypes such as Greece remain covered.
- The comparison is operational and annual: it does not include embodied carbon of retrofit materials or replacement systems.
- Results depend on the selected weather source, scenario definitions, and UNI/TS 11300 configuration used during the run.
