from MLopsProject import logger
from MLopsProject.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from MLopsProject.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from MLopsProject.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from MLopsProject.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed<<<<<<<<\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataValidationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = DataTransformationTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n")
except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainerTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<<<<<<<\n")
except Exception as e:
        logger.exception(e)
        raise e