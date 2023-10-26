setup: Celeb-real Celeb-synthesis frame_extractor.py srm_script.py
	python frame_extractor.py Celeb-real real_images 320 10
	python frame_extractor.py Celeb-synthesis fake_images 320 10
	python srm_script.py real_images real_images_srm
	python srm_script.py fake_images fake_images_srm

train: real_images_srm real_images_ela fake_images_srm fake_images_ela
	python train.py real_images_srm fake_images_srm srm_model.h5 srm_model
	python train.py real_images_ela fake_images_ela ela_model.h5 ela_model

benchmark: models/ela_model.h5 models/srm_model.h5 ela_model_test_db srm_model_test_db
	python benchmark.py