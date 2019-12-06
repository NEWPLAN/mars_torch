all:
	echo "which do you want to run? choose from : [train, retrain, test]"

train:
	python main.py train --cuda=1 --MAX_STEP=5000
retrain:
	python main.py retrain --CHECKPOINT_START_EPISODE=800 --cuda=1
test:
	python main.py test