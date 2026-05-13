
import os
from src.altering_plate_multiple_collection_methods import alter_scanner_files_MAD_bridging as alter_multiple_mad


def alter_plates_for_bridging(plates, plate_path, param_path_dict, plate_save_path, median_only=False):
    
    
    
    for plate in plates:
        print("###################################################################################################")

        alter_multiple_mad(f'{plate_path}/{plate}',
                f'{plate_path}/{plate}/{plate} Workbook.xlsx',
                
                param_path_dict,
                f'{plate_save_path}/{plate}',
                median_only=median_only)
        print("###################################################################################################")
        if median_only:
            print(f"Altered {plate} for median bridging")
        else:
            print(f"Altered {plate} for MAD bridging")

if __name__ == "__main__":
    
    plates = [f"OH2025_05{i}" for i in range(1, 8)]
    plates += [f"OH2026_00{i}" for i in range(1, 6)]
    plate_path = "data/plates/original_with_altered_workbooks"
    param_path_dict = {'Streck': 'data/params/pre_normalization/ds_log_16042026.json',
                    'NDS': 'data/params/pre_normalization/nds_log_16042026.json',}
    plate_save_path = "data/plates/altered_plates/MAD_bridging"
    
    alter_plates_for_bridging(plates, plate_path, param_path_dict, plate_save_path)
    
    plate_save_path_median = "data/plates/altered_plates/median_only_bridging"
    alter_plates_for_bridging(plates, plate_path, param_path_dict, plate_save_path_median, median_only=True)
