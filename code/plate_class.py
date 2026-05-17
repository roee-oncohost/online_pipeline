

import subprocess
import json
from pathlib import Path
# from projects.pre_post_bridging.datadelvetools_script import analyze_plate
from src.alter_plate import *


PROJECT_ROOT = Path(__file__).resolve().parents[1]

class plate_class:

    def __init__(self, plate_id):
        self.plate_id = plate_id

    

        
      
    def update_config_file(self, 
                    input_template_path, 
                    output_json_path, 
                    adat_path,
                    somamer_reference_source,
                    workdir_path,                    
                    site_id='',
                    study_id='', 
                    lot='',
                    platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                    calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                    anml_qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt',
                    qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
                    anml_smp_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt'):


        """
        Update a SOMAscan configuration JSON file for a specific plate.

        This function loads a template configuration JSON, updates plate-specific
        paths, identifiers, reference sources, and analysis-step settings, then
        writes the modified configuration to a new JSON file.

        Parameters
        ----------
        input_template_path : str or pathlib.Path
            Path to the input JSON template file.

        output_json_path : str or pathlib.Path
            Path where the updated JSON configuration file should be written.

        adat_path : str
            Base output directory for ADAT results. If provided, the function sets
            '!OutputDirectory' to '<adat_path>/<plate_id>'.

        somamer_reference_source : str
            Path to the SOMAmer reference source file. If provided, updates
            '!SOMAmerReferenceSource'.

        workdir_path : str
            Base working directory containing plate-specific files. If provided,
            updates '!SampleWorkbooks' and '!AgilentDirectory' using `self.plate_id`.

        site_id : str, optional
            Site identifier to write to the 'SiteId' field.

        study_id : str, optional
            Study identifier to write to the '!StudyId' field.

        lot : str, optional
            Master mix lot identifier to write to the '!MasterMixLot' field.

        platescale_reference_source : str, optional
            Reference source used for analysis steps with stepType 'plateScale'.

        calibrate_reference_source : str, optional
            Reference source used for analysis steps with stepType 'calibrate'.

        anml_qc_reference_source : str, optional
            Reference source used for analysis steps with stepName 'anmlQC'.

        qc_reference_source : str, optional
            QC reference source used for analysis steps with stepType 'qcCheck'.

        anml_smp_reference_source : str, optional
            Reference source used for analysis steps with stepName 'anmlSMP'.

        Returns
        -------
        pathlib.Path
            Path to the updated JSON configuration file.

       
        Notes
        -----
        - The function relies on `self.plate_id` to construct plate-specific paths.
        - Only non-empty optional values are applied.
        - Analysis step updates are made in-place within
        `data["ReportConfig"]["analysisSteps"]`.
        """
        
        input_template_path = Path(input_template_path)
        output_json_path = Path(output_json_path)

        # Load JSON
        with open(input_template_path, 'r') as f:
            data = json.load(f)

        if site_id:
            data["SiteId"] = site_id
        if adat_path:
            data["!OutputDirectory"] = adat_path # os.path.join(adat_path, self.plate_id) 
        if somamer_reference_source:
            data["!SOMAmerReferenceSource"] = somamer_reference_source
        if workdir_path:
            data["!SampleWorkbooks"] = [os.path.join(workdir_path, self.plate_id, f"{self.plate_id} Workbook.xlsx")] 
            data["!AgilentDirectory"] = os.path.join(workdir_path, self.plate_id) 
            
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

        # Create parent directories for the output file path, not the file itself.
        output_json_path.parent.mkdir(parents=True, exist_ok=True)

        # Save updated JSON
        with open(output_json_path, 'w') as f:
            json.dump(data, f, indent=4)

        return output_json_path
    
    
    def transform_scanner_files(self, plate_path, workbook_path, param_dict_path, new_plate_path, hce_list = HCE_LIST, method='MAD'):
        print(f"Transforming scanner files for plate {self.plate_id} using method {method}")
        alter_scanner_files(plate_path, workbook_path, param_dict_path, new_plate_path, hce_list, method)
    
    
    
    
    def ingest_plate(self, datadelve_path, config_path):
        """
        Execute DataDelve ingestion for a specific plate using a generated config file.

        This function constructs and runs a shell command to ingest a plate into
        DataDelve, using a plate-specific configuration JSON. It reports success
        or failure based on the subprocess execution outcome.

        Parameters
        ----------
        datadelve_path : str
            Path to the DataDelve executable or CLI entry point.

        config_path : str
            Base directory containing plate-specific configuration folders.
            The function expects the config file at:
            '<config_path>/<plate_id>/V4.1_plasma_7KL1.json'.

        Returns
        -------
        bool
            True if the ingestion command completes successfully, False if it fails.

        Notes
        -----
        - The command is executed via `subprocess.run` with `shell=True` and
        `check=True`, so a non-zero exit code raises `CalledProcessError`.
        - Uses a fixed encryption key argument (`--encrypt-key 0`).
        - Relies on `self.plate_id` to locate the correct configuration file.
        - Errors are caught and printed, but not re-raised.
        """
        print(f"Ingesting plate {self.plate_id}")
        try:
            
            # config_file = os.path.join(config_path, self.plate_id, "V4.1_plasma_7KL1.json")
            subprocess.run(
                [str(datadelve_path), "ingest", str(config_path), "--encrypt-key", "0"],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error ingesting plate {self.plate_id}: {e}")
            return False
        


    def analyze_plate(self, datadelve_path, adat_path):
        """
        Execute DataDelve analysis for a specific plate.

        This function runs the DataDelve `analyze` command on a previously ingested
        plate, using the plate-specific ADAT output directory. It reports success
        or failure based on the subprocess execution outcome.

        Parameters
        ----------
        datadelve_path : str
            Path to the DataDelve executable or CLI entry point.

        adat_path : str
            Base directory containing ADAT outputs. The function expects plate data at:
            '<adat_path>/<plate_id>'.

        Returns
        -------
        bool
            True if the analysis command completes successfully, False if it fails.

        Notes
        -----
        - The command executed is:
            '{datadelve_path} analyze {adat_path}/{plate_id} {plate_id} --encrypt-key 0'
        - Executed via `subprocess.run` with `shell=True` and `check=True`; a non-zero
        exit code raises `CalledProcessError`.
        - Uses a fixed encryption key argument (`--encrypt-key 0`).
        - Relies on `self.plate_id` to locate inputs and name the analysis run.
        - Errors are caught and printed, but not re-raised.
        """
        print(f"Analyzing plate {self.plate_id}")
        try:
           
            # plate_adat_path = os.path.join(adat_path, self.plate_id)
            subprocess.run(
                [str(datadelve_path), "analyze", str(adat_path), self.plate_id, "--encrypt-key", "0"],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error analyzing plate {self.plate_id}: {e}")
            return False
        
    def export_plate(self, datadelve_path, adat_path):
        """
        Execute DataDelve export for a specific plate.

        This function runs the DataDelve `export` command to generate output files
        for a previously analyzed plate, using the plate-specific ADAT directory.
        It reports success or failure based on the subprocess execution outcome.

        Parameters
        ----------
        datadelve_path : str
            Path to the DataDelve executable or CLI entry point.

        adat_path : str
            Base directory containing ADAT outputs. The function expects plate data at:
            '<adat_path>/<plate_id>'.

        Returns
        -------
        bool
            True if the export command completes successfully, False if it fails.

        Notes
        -----
        - The command executed is:
            '{datadelve_path} export {adat_path}/{plate_id} {plate_id} --encrypt-key 0'
        - Executed via `subprocess.run` with `shell=True` and `check=True`; a non-zero
        exit code raises `CalledProcessError`.
        - Uses a fixed encryption key argument (`--encrypt-key 0`).
        - Relies on `self.plate_id` to locate inputs and name the export.
        - Errors are caught and printed, but not re-raised.
        """
        print(f"Exporting plate {self.plate_id}")
        try:
            
            # plate_adat_path = os.path.join(adat_path, self.plate_id)
            subprocess.run(
                [str(datadelve_path), "export", str(adat_path), self.plate_id, "--encrypt-key", "0"],
                check=True,
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error exporting plate {self.plate_id}: {e}")
            return False

    def adat_generation(self, datadelve_path, output_path, adat_path):
        """
        Execute the full DataDelve processing pipeline for a single plate.

        This function orchestrates the sequential execution of ingestion, analysis,
        and export steps for a plate. Each step must succeed for the pipeline to
        continue; failure at any stage halts execution and reports an overall failure.

        Parameters
        ----------
        datadelve_path : str
            Path to the DataDelve executable or CLI entry point.

        output_path : str
            Path to the directory containing plate-specific configuration files,
            used during the ingestion step.

        adat_path : str
            Path to the directory containing ADAT data and outputs, used during
            analysis and export steps.

        plate : str
            Plate identifier. (Note: currently unused; the function relies on
            `self.plate_id`.)

        Returns
        -------
        bool
            True if all steps (ingest → analyze → export) complete successfully,
            False otherwise.

        Notes
        -----
        - Execution order is strictly:
            1. ingest_plate
            2. analyze_plate
            3. export_plate
        - Each step must return True for the next step to run.
        - Relies on `self.plate_id` internally rather than the `plate` argument.
        - Prints status messages indicating success or failure of the pipeline.
        """
        if self.ingest_plate(datadelve_path, output_path):
            print("done ingesting, moving to analyzing")
            if self.analyze_plate(datadelve_path, adat_path):
                print("done analyzing, moving to exporting")
                if self.export_plate(datadelve_path, adat_path):
                    print(f"Successfully processed plate {self.plate_id}")
                    return True
        print(f"Failed to process plate {self.plate_id}")
        return False
            
if __name__ == "__main__":
    print("Testing plate_class")
    
    workdir_path=f"somalogic/workdir/original_with_altered_workbooks"
    new_workdir_path = f"somalogic/workdir/lr"
    
    plates = [f'OH2025_05{num}' for num in range(1, 8)]
    plates += [f'OH2026_00{num}' for num in range(1, 6)]
    for plate_id in plates:
        output_json_path=f'somalogic/output/test/{plate_id}/V4.1_plasma_7KL1.json'
        adat_path=f'somalogic/adat/test/lr/roee_params/{plate_id}'
        if plate_id:# == 'OH2025_054':
            
            plate = plate_class(plate_id)
            plate.transform_scanner_files(os.path.join(workdir_path, plate_id), 
                                        os.path.join(workdir_path, plate_id, f"{plate_id} Workbook.xlsx"), 
                                        {'Streck': 'data/params/pre_normalization/ds_lr_roee_14052026.json',
                                         'NDS': 'data/params/pre_normalization/nds_lr_roee_14052026.json'},                                        
                                        # {'ignore 1': 'data/params/pre_normalization/ds_lr_30042026.json',
                                        #  'ignore 2': 'data/params/pre_normalization/nds_lr_30042026.json'},
                                        os.path.join(new_workdir_path, plate_id), method='linear_regression')
            plate.update_config_file( 
                            input_template_path='somalogic/output/lot1_prehybridization_median/OH2025_053/V4.1_plasma_7KL1.json', 
                            output_json_path=output_json_path, 
                            adat_path=adat_path,
                            somamer_reference_source="references/7KL1/SD4.1ReV_7K_Annotated_SOMAmers.xlsx",
                            workdir_path=new_workdir_path,                    
                            site_id=f'{plate_id}',
                            study_id=f'{plate_id}', 
                            lot='1',
                            platescale_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                            calibrate_reference_source='references/7KL1/plasma/Reference_Cal_200169_Plasma_V4.1_Lot1.txt',
                            anml_qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt',
                            qc_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_QC_ANML_200170.txt',
                            anml_smp_reference_source='references/7KL1/plasma/Reference_V4.1_Plasma_ANML.txt')


            # plate.transform_scanner_files(os.path.join('somalogic/workdir/test', plate_id),)
            # plate.adat_generation('somalogic/datadelvetools', output_json_path, adat_path=adat_path)            
    
   