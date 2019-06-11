import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from visualize import show_spectrogram, show_attention
import src.Tacotron_model.taco_util as util



from src.Tacotron_model.taco_util import sequence_to_text
from src.dataset import collate_fn

def train(model, optimizer, dataset, num_epochs, batch_size=1, log_interval=50):
        model.train()

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        step = 0
        for epoch in tqdm(range(hparams.num_epochs), total=hparams.num_epochs, unit=' epochs'):
            total_loss = 0
            pbar = tqdm(loader, total=len(loader), unit=' batches')

            for b, (text_batch, audio_batch, text_lengths, audio_lengths) in enumerate(pbar):
                text = Variable(text_batch).cpu()
                targets = Variable(audio_batch, requires_grad=False).cpu()


                #  create stop targets
                stop_targets = torch.zeros(targets.size(1), targets.size(0))

                for i in range(len(stop_targets)):
                    stop_targets[i, audio_lengths[i] - 1] = 1
                stop_targets = Variable(stop_targets, requires_grad=False).cpu()
                outputs, stop_tokens, attention = model(text, targets)
                spec_loss = F.mse_loss(outputs, targets)
                stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
                loss = spec_loss + stop_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.data[0]
                pbar.set_description(f'loss: {loss.data[0]:.4f}')
                #writer.add_scalar('loss', loss.data[0], step)
                if step % log_interval == 0:
                    torch.save(model.state_dict(), '/home/rajanie/Documents/Semester2/TTS/tts/models/melnet_{step}.pt')

                    # plot the first sample in the batch

                step += 1

        attention_plot = show_attention(attention[0], return_array=True)
        output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                       sequence_to_text(text.data[0]),
                                       return_array=True)
        target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                       sequence_to_text(text.data[0]),
                                       return_array=True)


def val(model, dataset, num_epochs, batch_size=1, log_interval=50, hparams):
    model.val()

    loader = DataLoader(dataset, batch_size=hparams.batch_size, collate_fn=collate_fn)
    step = 0
    for epoch in tqdm(range(hparams.num_epochs), total=hparams.num_epochs, unit=' epochs'):
        total_loss = 0
        pbar = tqdm(loader, total=len(loader), unit=' batches')

        for b, (text_batch, audio_batch, text_lengths, audio_lengths) in enumerate(pbar):
            text = Variable(text_batch).cpu()
            targets = Variable(audio_batch, requires_grad=False).cpu()

            #  create stop targets
            stop_targets = torch.zeros(targets.size(1), targets.size(0))

            for i in range(len(stop_targets)):
                stop_targets[i, audio_lengths[i] - 1] = 1
            stop_targets = Variable(stop_targets, requires_grad=False).cpu()
            outputs, stop_tokens, attention = model(text, targets)
            spec_loss = F.mse_loss(outputs, targets)
            stop_loss = F.binary_cross_entropy_with_logits(stop_tokens, stop_targets)
            loss = spec_loss + stop_loss
            total_loss += loss.data[0]
            pbar.set_description(f'loss: {loss.data[0]:.4f}')
            # writer.add_scalar('loss', loss.data[0], step)
            if step % log_interval == 0:
                torch.save(model.state_dict(), '/home/rajanie/Documents/Semester2/TTS/tts/models/melnet_{step}.pt')

                # plot the first sample in the batch

            step += 1

    attention_plot = show_attention(attention[0], return_array=True)
    output_plot = show_spectrogram(outputs.data.permute(1, 2, 0)[0],
                                   sequence_to_text(text.data[0]),
                                   return_array=True)
    target_plot = show_spectrogram(targets.data.permute(1, 2, 0)[0],
                                   sequence_to_text(text.data[0]),
                                   return_array=True)
	avg_loss = torch.mean(total_loss)

	print("Step: {} Loss: {}".format(step, avg_loss))

	return avg_loss


def train_and_evaluate(dataset, hparams):
	model = util.fetch_model(hparams)
	dataloader = util.fetch_dataloader(dataset, model, hparams)
	optimizer = util.fetch_optimizer(model, hparams)

	global_step = 0
	best_metric = 0.0

	for epoch in range(hparams.num_epoch):

		print("Epoch: {}".format(epoch))

		global_step = train(dataloader['train'], model, optimizer, global_step, hparams)

		metric = val(dataloader['val'], model, global_step, hparams)

		is_best = False
		if metric >= best_metric:
			is_best = True
			best_metric = metric

		if is_best:
			print("yey")