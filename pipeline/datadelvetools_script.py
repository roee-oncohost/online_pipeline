import subprocess
import json




import json
from pathlib import Path

def update_config_file(input_template_path, 
                   output_json_path, 
                    adat_path,
                    somamer_reference_source,
                    workdir_path, 
                    plate, 
                    site_id='',
                    study_id='', 
                    lot='',
                    platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                    calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                    anml_qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt',
                    qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
                    anml_smp_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt'):
    """
    Load a JSON file, update the 'SiteId' field, and save to a new file.

    Args:
        input_json_path (str or Path): Path to the original JSON file
        output_json_path (str or Path): Path to save the modified JSON
        new_site_id (str): New value for 'SiteId'
    """
    input_template_path = Path(input_template_path)
    output_json_path = Path(output_json_path)

    # Load JSON
    with open(input_template_path, 'r') as f:
        data = json.load(f)

    # Update SiteId
    if "SiteId" not in data:
        raise KeyError("SiteId not found in JSON")

    if site_id:
        data["SiteId"] = site_id
    if adat_path:
        data["!OutputDirectory"] = f"{adat_path}/{plate}"
    if somamer_reference_source:
        data["!SOMAmerReferenceSource"] = somamer_reference_source
    if workdir_path:
        data["!SampleWorkbooks"] = [f"{workdir_path}/{plate}/{plate} Workbook.xlsx"]
        data["!AgilentDirectory"] = f"{workdir_path}/{plate}/"
    if study_id:
        data["!StudyId"] = study_id
    if lot:
        data["!MasterMixLot"] = lot
    if platescale_reference_source:
        for step in data["ReportConfig"]["analysisSteps"]:
            if step.get("stepType", "N/A") == "plateScale":
                step["referenceSource"] = platescale_reference_source
    if calibrate_reference_source:
        for step in data["ReportConfig"]["analysisSteps"]:
            if step.get("stepType", "N/A") == "calibrate":
                step["referenceSource"] = calibrate_reference_source
    if anml_qc_reference_source:
        for step in data["ReportConfig"]["analysisSteps"]:
            if step.get("stepName", "N/A") == "anmlQC":
            # if step["stepName"] == "anmlQC":
                step["referenceSource"] = anml_qc_reference_source
    if qc_reference_source:
        for step in data["ReportConfig"]["analysisSteps"]:
            if step.get("stepType", "N/A") == "qcCheck":
                step["QCReferenceSource"] = qc_reference_source
    if anml_smp_reference_source:
        for step in data["ReportConfig"]["analysisSteps"]:
            if step.get("stepName", "N/A") == "anmlSMP":
                step["referenceSource"] = anml_smp_reference_source

    # Save updated JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    return output_json_path


def ingest_plate(datadelve_path, output_path, plate):
    print(f"Ingesting plate {{plate}}")
    try:
        subprocess.run(f"{{datadelve_path}} ingest {{output_path}}/{{plate}}/V4.1_plasma_7KL1.json --encrypt-key 0", shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error ingesting plate {{plate}}: {{e}}")
        return False

def analyze_plate(datadelve_path, adat_path, plate):
    print(f"Analyzing plate {{plate}}")
    try:
        subprocess.run(f"{{datadelve_path}} analyze {{adat_path}}/{{plate}} {{plate}} --encrypt-key 0", shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error analyzing plate {{plate}}: {{e}}")
        return False
    
def export_plate(datadelve_path, adat_path, plate):
    print(f"Exporting plate {{plate}}")
    try:
        subprocess.run(f"{{datadelve_path}} export {{adat_path}}/{{plate}} {{plate}} --encrypt-key 0", shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error exporting plate {{plate}}: {{e}}")
        return False
    
def run_plate_processing(datadelve_path, output_path, adat_path, plate):
    if ingest_plate(datadelve_path, output_path, plate):
        if analyze_plate(datadelve_path, adat_path, plate):
            if export_plate(datadelve_path, adat_path, plate):
                print(f"Successfully processed plate {{plate}}")
                return True
    print(f"Failed to process plate {{plate}}")
    return False


def plate_full_run(
    update_config=True,
    pre_normalization_bridging=True,
    run_plate=True,
    config_template_path='', 
    config_output_path='', 
    adat_path='',
    somamer_reference_source='',
    workdir_path='', 
    plate='', 
    site_id='',
    study_id='', 
    lot='',
    platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
    calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
    anml_qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt',
    qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
    anml_smp_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt'):
    update_config_file(config_template_path, 
                   config_output_path, 
                   adat_path,
                   somamer_reference_source,
                   workdir_path, 
                   plate, 
                   site_id='OH2025_001',
                   study_id='OH2025_001', 
                   lot='',
                   platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                   calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                   anml_qc_reference_source='',
                   qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
                   anml_smp_reference_source='KOWABANGA!!!')
    
    
    pass


if __name__ == "__main__":
    print("strated")
    datadelve_path = 'somalogic/datadelvetools'
    output_path = 'somalogic/output/lot1_prehybridization_mad'
    adat_path = 'somalogic/adat/mad_transformed_29042026'
    ingest_plate(datadelve_path, output_path, plate)
    plates = ['OH2025_051']
    # for plate in plates:
    #     run_plate_processing(datadelve_path, output_path, adat_path, plate)
    # create_output_file('hellllllllo!!!!!',
    #                    '',
    #                    '', 
    #                    '', 
    #                    study_id='', 
    #                    lot='Lot1',
    #                    )
    
    input_template_path = 'somalogic/output/lot1_prehybridization_mad/OH2025_051/V4.1_plasma_7KL1.json'
    output_template_path = 'somalogic/output_result.json'
    somamer_reference_source = 'Yehonatan!!!'
    workdir_path = 'Gil!!!'
    plate = plates[0]
    update_config_file(input_template_path, 
                   output_template_path, 
                   adat_path,
                   somamer_reference_source,
                   workdir_path, 
                   plate, 
                   site_id='OH2025_001',
                   study_id='OH2025_001', 
                   lot='',
                   platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                   calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                   anml_qc_reference_source='',
                   qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
                   anml_smp_reference_source='KOWABANGA!!!')