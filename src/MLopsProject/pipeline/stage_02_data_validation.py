from MLopsProject.config.configuration import ConfigurationManager
from MLopsProject.components.data_validation import DataValidation
from MLopsProject import logger

STAGE_NAME = "DATA VALIDATION STAGE"

class DataValidationTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE_NAME} ended <<<<<<<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e
