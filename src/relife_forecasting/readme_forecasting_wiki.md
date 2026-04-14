# ReLIFE Forecasting Service

## Introduction

The ReLIFE Forecasting Service is a comprehensive simulation-based platform designed to assess the energy and indoor comfort performance of individual buildings and building stocks. Developed within the framework of the European EPBD Recast directive, the service leverages the ISO 52016-1 and ISO 52010 calculation procedures to support evidence-based renovation planning up to 2030 and 2050 horizons.

The service enables stakeholders to evaluate building energy performance, enhance environmental sustainability, increase economic value, reduce energy consumption and CO₂ emissions, and maximize the integration of Renewable Energy Sources (RES) through dynamic simulation scenarios.

## Core Objectives

- **Performance Assessment**: Evaluate energy and indoor comfort performance of buildings using dynamic simulation
- **Renovation Planning**: Support evidence-based renovation strategies with long-term horizons (2030-2050)
- **Sustainability Enhancement**: Reduce energy consumption and CO₂ emissions through optimized interventions
- **RES Integration**: Maximize renewable energy integration through photovoltaic and other clean technologies
- **Climate Resilience**: Assess building performance under future climate change scenarios

## Technical Implementation

### Computational Engine

The service is built on the **pyBuildingEnergy** Python library as its core computational engine, implementing:

- **ISO 52016-1**: Dynamic heating and cooling energy needs calculation
- **ISO 52010**: Solar irradiance and daylight availability assessment
- **UNI/TS 11300**: Primary energy and system performance evaluation

### Energy Conservation Measures (ECMs)

The system supports comprehensive ECM evaluation including:

#### Envelope Interventions
- **Roof Insulation**: Thermal transmittance optimization for horizontal opaque surfaces
- **Wall Insulation**: External/internal insulation for vertical opaque surfaces  
- **Window Replacement**: High-performance glazing and frame systems
- **Slab Insulation**: Ground floor thermal improvement

#### System-Level Upgrades
- **Heat Pump Integration**: Electric heat pump systems with configurable COP
- **Condensing Boiler**: High-efficiency gas combustion systems
- **Hybrid Systems**: Combined renewable and conventional systems

#### Renewable Energy Solutions
- **Photovoltaic Systems**: On-site electricity generation with PVGIS integration
- **Solar Thermal**: Hot water generation from solar collectors

## Climate Scenarios Integration

A key innovation of the ReLIFE Forecasting Service is its capability to incorporate **future climate scenarios** through projected weather files:

### Future Weather Data
- **Typical Meteorological Year (TMY)** datasets for future climate projections
- **Climate Change Trajectories**: Multiple scenarios for 2030 and 2050 horizons
- **Long-term Resilience Assessment**: Evaluation of renovation measures under evolving climatic conditions

### Climate Impact Analysis
- **Temperature Evolution**: Impact on heating/cooling demand patterns
- **Weather Extremes**: Building performance under extreme weather events
- **Adaptation Strategies**: Climate-resilient renovation planning

## Scenario-Based Evaluation

The service enables comprehensive scenario analysis through:

### Multi-Scenario Comparison
- **Baseline Assessment**: Current building performance evaluation
- **Individual ECMs**: Single intervention impact analysis
- **Combined Measures**: Multiple ECM interaction effects
- **System Variants**: Different heating/cooling system configurations

### Performance Metrics
- **Energy Needs**: Heating and cooling demand (kWh/year)
- **Primary Energy**: Total energy consumption including system losses
- **CO₂ Emissions**: Environmental impact assessment
- **Comfort Indicators**: Indoor temperature and humidity analysis
- **Economic Indicators**: Cost-benefit analysis integration

## API Architecture

### Core Endpoints

#### Building Simulation
- **POST /simulate**: Single building energy simulation
- **POST /simulate/batch**: Parallel multi-building simulation
- **POST /ecm_application**: Envelope retrofit scenario analysis

#### Energy Performance
- **POST /primary-energy/uni11300**: Primary energy calculation
- **POST /validate**: Building model validation

#### CO₂ Analysis
- **POST /calculate**: Single scenario CO₂ emissions
- **POST /compare**: Multi-scenario emission comparison
- **POST /calculate-intervention**: Retrofit intervention impact

#### Reporting
- **POST /report**: HTML energy analysis report generation
- **GET /docs**: Interactive API documentation

### Input Modes

#### Archetype Mode
- Predefined building templates by category, country, and construction period
- Standardized system configurations for typical building types
- Rapid scenario evaluation for building stock analysis

#### Custom Mode
- User-defined building models via BUI (Building Unit Input) JSON
- Custom HVAC system configurations
- Flexible integration with external modeling tools

## Data Requirements

### Technical Building Data
- **Geometry**: Floor area, volume, surface orientations
- **Construction**: U-values, thermal capacity, solar absorptance
- **Systems**: Heating/cooling equipment, distribution networks
- **Occupancy**: Internal gains, ventilation rates, setpoint temperatures

### Climate Data
- **Current Weather**: PVGIS real-time meteorological data
- **EPW Files**: EnergyPlus weather data formats
- **Future Projections**: Climate change scenario datasets

### Energy Parameters
- **Emission Factors**: Country-specific CO₂ emission coefficients
- **Energy Prices**: Market-based energy cost projections
- **System Efficiencies**: Equipment performance characteristics

## Methodology Framework

### Simulation Workflow

1. **Model Preparation**: Building geometry and construction validation
2. **Climate Processing**: Weather data preparation and interpolation
3. **Energy Calculation**: ISO 52016 dynamic simulation
4. **System Analysis**: HVAC system performance evaluation
5. **Results Processing**: Data aggregation and report generation

### Quality Assurance

- **Model Validation**: BUI schema compliance checking
- **Result Verification**: Energy balance and physical consistency checks
- **Error Handling**: Comprehensive exception management and reporting
- **Performance Monitoring**: Computational efficiency optimization

## Integration Capabilities

### External Systems
- **Financial Service**: Economic analysis integration
- **GIS Platforms**: Spatial building stock mapping
- **BIM Tools**: Building information modeling interoperability
- **Energy Management Systems**: Real-time monitoring integration

### Data Formats
- **JSON API**: RESTful web service interface
- **CSV Export**: Tabular data for spreadsheet analysis
- **HTML Reports**: Interactive visualization dashboards
- **BUI Schema**: Standardized building model format

## Applications

### Stakeholder Groups

#### Professionals
- **Energy Consultants**: Detailed building performance analysis
- **Architects**: Design optimization and compliance checking
- **Engineers**: System sizing and integration planning

#### Policy Makers
- **Public Administrations**: Building stock renovation planning
- **Regulatory Bodies**: Energy code compliance verification
- **Urban Planners**: District-scale energy strategies

#### Property Owners
- **Building Managers**: Operational optimization
- **Homeowners**: Renovation decision support
- **Investors**: Property valuation enhancement

### Use Cases

- **Renovation Planning**: Optimal ECM selection and sequencing
- **Compliance Verification**: Energy regulation adherence checking
- **Portfolio Analysis**: Multi-building performance assessment
- **Climate Adaptation**: Future-proofing building investments

## Future Development

### Roadmap Items
- **Machine Learning Integration**: Enhanced prediction capabilities
- **Advanced Visualization**: 3D modeling and augmented reality
- **Real-time Monitoring**: IoT sensor integration
- **Blockchain Integration**: Energy trading and carbon credits

### Research Directions
- **Urban Scale Modeling**: District heating and cooling optimization
- **Material Innovation**: New construction technologies evaluation
- **Behavioral Analysis**: Occupant interaction patterns
- **Circular Economy**: Material lifecycle and reuse assessment

---

## Authors and Reviewers

**Developed by**: Decision Support Systems Laboratory (DSS Lab) at the National Technical University of Athens (NTUA)

**Core Team**:
- Building Energy Simulation Specialists
- Software Architecture Engineers
- Climate Science Researchers
- Energy Policy Analysts

## License

The ReLIFE Forecasting Service is licensed under the **EUPL-1.2** license, ensuring open-source accessibility while maintaining intellectual property protection.

## Acknowledgement

This work is carried out within the ReLIFE project and is co-funded by the European Union (CINEA) under Grant Agreement No. 101167067.

*Views and opinions expressed are those of the authors and do not necessarily reflect those of the European Union or CINEA. Neither the European Union nor CINEA can be held responsible for them.*
