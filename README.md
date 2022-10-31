# DNN midterm assignments

VGG-Net 을 이용한 batch size, learning rate, model depth 등의 hyper-parameter 변화를 통한 CIFAR-10에서의 classification 성능 확인

- Training 
	> python run_training.py
	
- Change batch size (default : 128)
	> python run_training.py --batch_size []

- Change learning size (default : 1e-7)
	> python run_training.py --learning_rate []

- Change model (default: vgg11)
	> python run_training.py --model_name vgg16
