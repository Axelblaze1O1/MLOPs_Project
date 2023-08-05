from MLopsProject.config.configuration import ConfigurationManager
from MLopsProject.components.model_trainer import ModelTrainer
from MLopsProject import logger
from pathlib import Path

STAGE_NAME = "Model Training Stage"

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        Trainer_obj = ModelTrainer(config=model_trainer_config)
        Trainer_obj.train()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<<<<")
        obj = ModelTrainerTrainingPipeline()
        obj.main()
        logger.info(f">>>>>>>> stage {STAGE_NAME} completed <<<<<<<<<\n")
    except Exception as e:
        logger.exception(e)
        raise e