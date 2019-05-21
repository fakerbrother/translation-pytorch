from DataSet import *
from seq2seq_model import *
import random
import time
import math
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
plt.switch_backend('agg')


teacher_forcing_ratio = 0.5
save_dir = os.path.join(dir_path, "model")


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0.0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return "{}m {}s".format(m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return "{} (- {})".format(asMinutes(s), asMinutes(rs))


def train_iters(input_lang, output_lang, pairs, max_length, encoder, decoder,
                n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensor_from_pair(input_lang, output_lang, random.choice(pairs), device) for _ in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters+1):
        training_pair = training_pairs[iter-1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer,
                     decoder_optimizer, criterion, max_length)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print("{} ({} {}%) {:.4f}".format(timeSince(start, iter/n_iters), iter, iter/n_iters*100, print_loss_avg))
            directory = os.path.join(save_dir, "{}_checkpoint.tar".format(iter/print_every))
            torch.save({
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'in_lang': input_lang.__dict__,
                'out_lang': output_lang.__dict__,
                'max_len': max_length
            }, directory)

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


if __name__ == "__main__":
    hidden_size = 256
    input_lang, output_lang, pairs, max_length = prepareData()
    encoder = EncodeRnn(input_lang.n_words, hidden_size).to(device)
    attn_encoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1, max_length=max_length).to(device)
    train_iters(input_lang, output_lang, pairs, max_length, encoder, attn_encoder, n_iters=75000, print_every=5000)
