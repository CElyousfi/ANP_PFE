# rag-service/create_test_documents.py
import os
import argparse

def create_port_regulation_document(output_path):
    """
    Create a simple port regulation document for testing.
    """
    content = """# Port Regulations and Guidelines

## Article 1: General Port Information
The port operates 24/7 with specific loading and unloading schedules as determined by port authorities.
All vessels must register at least 24 hours before arrival at any Moroccan port.
Port authorities reserve the right to modify schedules based on weather conditions and operational requirements.

## Article 2: Safety Regulations
1. All personnel must wear appropriate safety gear in designated areas including helmets, high-visibility vests, and safety boots.
2. Emergency procedures must be clearly displayed on all vessels and in all port facilities.
3. Regular safety drills are conducted monthly to ensure preparedness for emergencies.
4. Vessels carrying hazardous materials must display appropriate warning signs and notify port authorities in advance.
5. Speed limit within port waters is 5 knots unless otherwise specified.

## Article 3: Environmental Policies
The port follows strict environmental guidelines to minimize pollution:
- All waste disposal must follow established protocols.
- Ballast water management plans must be submitted and approved.
- Oil spill prevention measures must be in place for all vessels.
- Air emissions must meet national and international standards.
- Noise levels should be kept to a minimum, especially during night operations.

## Article 4: Commercial Operations
Commercial operations at the port are governed by the following principles:
1. Cargo handling charges are based on the current tariff schedule.
2. Storage fees apply after the first 48 hours of cargo remaining in port.
3. Priority berthing may be arranged for regular service vessels.
4. All commercial disputes shall be resolved according to Moroccan maritime law.
5. Payment must be secured before cargo release.

## Article 5: Security Measures
Security is paramount within all port areas:
- All persons entering port facilities must have proper identification.
- CCTV monitoring is in operation throughout the port.
- Random security checks may be performed at any time.
- Restricted areas require special authorization for access.
- All suspicious activities must be reported immediately to port security.

## Contact Information
Port Authority Office: +212-5XX-XXXXXX
Emergency Contact: +212-5XX-XXXXXX
Email: info@anp.org.ma
Website: www.anp.org.ma
"""

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created test document: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating test document: {e}")
        return False

def create_technical_document(output_path):
    """
    Create a technical document about port equipment.
    """
    content = """# Port Equipment Technical Specifications

## 1. Crane Specifications
### 1.1 Ship-to-Shore (STS) Cranes
- Lifting capacity: 65 tons under spreader
- Outreach: 65 meters
- Lifting height: 42 meters above rail
- Lifting speed: 90 meters/min with load
- Trolley speed: 180 meters/min
- Gantry speed: 45 meters/min
- Power supply: 11kV/60Hz

### 1.2 Rubber Tyred Gantry (RTG) Cranes
- Lifting capacity: 40 tons under spreader
- Lifting height: 18.5 meters (6+1 containers)
- Span: 23.6 meters
- Lifting speed: 30 meters/min with load
- Trolley speed: 70 meters/min
- Gantry speed: 130 meters/min
- Power: Diesel-electric or electric with cable reel

## 2. Handling Equipment
### 2.1 Reach Stackers
- Lifting capacity: 45 tons in first row
- Stacking capability: 5 containers high
- Engine power: 350 HP
- Maximum speed: 25 km/h

### 2.2 Terminal Tractors
- Engine power: 260 HP
- Maximum speed: 40 km/h
- Turning radius: 6.3 meters
- Coupling height: 1150 mm

## 3. Maintenance Requirements
All port equipment must be maintained according to the following schedule:
- Daily inspections: Before each shift
- Weekly maintenance: Lubrication and minor adjustments
- Monthly maintenance: Comprehensive systems check
- Quarterly maintenance: Major systems overhaul
- Annual certification: Safety and capacity verification

Maintenance records must be kept for a minimum of 5 years and be available for inspection at any time.

## 4. Safety Systems
All equipment must include the following safety features:
- Emergency stop buttons
- Overload warning systems
- Anti-collision systems
- Wind speed monitoring with automatic shutdown
- Operator presence sensors
- Audible movement alarms
- Comprehensive lighting for night operations

## 5. Environmental Specifications
- Emissions standards: Euro Stage V or equivalent
- Noise level: <82 dB at operator position
- Energy efficiency: Smart power management systems
- Fuel consumption monitoring
- Biodegradable hydraulic fluids

## 6. Training Requirements
Operators must be certified after completing:
- 40 hours theoretical training
- 80 hours practical operation under supervision
- Safety procedures examination
- Equipment-specific certification
- Annual refresher courses
"""

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created technical document: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating technical document: {e}")
        return False

def create_commercial_document(output_path):
    """
    Create a commercial document about port fees and tariffs.
    """
    content = """# Port Commercial Tariffs and Procedures

## Section 1: Vessel-Related Charges
### 1.1 Port Dues
| Vessel Size (GRT) | Rate (MAD per GRT) |
|-------------------|-------------------|
| 0 - 500           | 3.50              |
| 501 - 1,000       | 4.25              |
| 1,001 - 5,000     | 5.00              |
| 5,001 - 10,000    | 5.75              |
| 10,001 - 20,000   | 6.50              |
| > 20,000          | 7.25              |

Minimum charge: 1,000 MAD per call
Special rates apply for regular service vessels with minimum 12 calls per year

### 1.2 Berth Occupancy Charges
| Duration          | Rate (MAD per meter of vessel length per hour) |
|-------------------|--------------------------------------------|
| First 24 hours    | 2.50                                       |
| 24 - 48 hours     | 3.00                                       |
| 48 - 96 hours     | 4.00                                       |
| > 96 hours        | 5.50                                       |

### 1.3 Pilotage Fees
Compulsory for all vessels over 100 GRT
| Vessel Size (GRT) | Rate (MAD) |
|-------------------|------------|
| 0 - 500           | 750        |
| 501 - 1,000       | 1,250      |
| 1,001 - 5,000     | 2,000      |
| 5,001 - 10,000    | 3,000      |
| 10,001 - 20,000   | 4,250      |
| > 20,000          | 5,500      |

### 1.4 Towage Services
Compulsory for all vessels over 500 GRT
| Vessel Size (GRT) | Rate per Tug (MAD per hour) |
|-------------------|-----------------------------|
| 0 - 1,000         | 2,000                       |
| 1,001 - 5,000     | 3,500                       |
| 5,001 - 15,000    | 5,000                       |
| 15,001 - 30,000   | 7,500                       |
| > 30,000          | 10,000                      |

Minimum charge: 1 hour per tug

## Section 2: Cargo-Related Charges
### 2.1 Cargo Handling
| Cargo Type           | Rate (MAD per ton) |
|----------------------|-------------------|
| General cargo        | 45                |
| Containerized cargo  | 35                |
| Bulk dry cargo       | 30                |
| Liquid bulk          | 25                |
| Ro-Ro (per vehicle)  | 200               |

### 2.2 Storage Charges
| Cargo Type     | Free Period | First Period (MAD/ton/day) | Second Period (MAD/ton/day) |
|----------------|-------------|----------------------------|----------------------------|
| General cargo  | 5 days      | 10 (days 6-15)             | 20 (after day 15)          |
| Containers     | 7 days      | 150 (days 8-15)            | 300 (after day 15)         |
| Bulk cargo     | 3 days      | 5 (days 4-10)              | 15 (after day 10)          |

### 2.3 Container Handling Charges
| Service                  | 20' Container (MAD) | 40' Container (MAD) |
|--------------------------|---------------------|---------------------|
| Discharge/Loading        | 750                 | 1,125               |
| Shifting on board        | 375                 | 560                 |
| Gate in/out              | 250                 | 375                 |
| Reefer connection (daily)| 300                 | 450                 |
| IMDG surcharge           | +50%                | +50%                |

## Section 3: Administrative Procedures
### 3.1 Required Documentation
- Vessel Pre-arrival Notice (72 hours)
- Customs Clearance Documentation
- Cargo Manifest
- Dangerous Goods Declaration (if applicable)
- Crew List and Passenger List (if applicable)
- Maritime Declaration of Health
- International Ship Security Certificate

### 3.2 Payment Terms
- All charges must be paid in Moroccan Dirhams (MAD)
- Payments due within 15 days of invoice date
- 2% late payment penalty per month
- Vessel agents must deposit sufficient funds before vessel departure
- Bank guarantees accepted from approved agents only

### 3.3 Dispute Resolution
Any dispute regarding port charges shall be submitted in writing to the Commercial Department within 30 days of invoice date.

For further information:
Commercial Department
Tel: +212-5XX-XXXXXX
Email: commercial@anp.org.ma
"""

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created commercial document: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating commercial document: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Create test documents for RAG service')
    parser.add_argument('--data-folder', type=str, default='data', help='Data folder root')
    args = parser.parse_args()
    
    # Create test documents in each department folder
    create_port_regulation_document(os.path.join(args.data_folder, 'regulatory', 'port_regulations.txt'))
    create_technical_document(os.path.join(args.data_folder, 'technical', 'equipment_specifications.txt'))
    create_commercial_document(os.path.join(args.data_folder, 'commercial', 'port_tariffs.txt'))
    
    print("Test documents created successfully. Use these to test the RAG service.")

if __name__ == "__main__":
    main()