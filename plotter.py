import matplotlib.pyplot as plt
plt.switch_backend('agg')

def plot(trains,valids):
	plt.plot(trains)
	plt.ylabel("train loss")
	plt.xlabel("epochs")
	plt.title("Train Losses")
	plt.savefig("./Contrastive-Predictive-Coding/train_loss.png",bbox_inches='tight')
	plt.clf()
	plt.plot(valids)
	plt.ylabel("validation loss")
	plt.xlabel("epochs")
	plt.title("Validation Losses")
	plt.savefig("./Contrastive-Predictive-Coding/validation_loss.png",bbox_inches='tight')
	
