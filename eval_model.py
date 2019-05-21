from seq2seq_model import *
from DataSet import *


def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length):
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        input_tensor = tensor_from_sentence(input_lang, sentence, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.init_hidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di+1]


def load_model(save_dir):
    input_lang = Lang("en")
    output_lang = Lang("cn")
    checkpoint = torch.load(save_dir)
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    input_lang.__dict__ = checkpoint['in_lang']
    output_lang.__dict__ = checkpoint['out_lang']
    max_length = checkpoint['max_len']
    encoder = EncodeRnn(input_lang.n_words, hidden_size=256)
    decoder = AttnDecoderRNN(hidden_size=256, output_size=output_lang.n_words, max_length=max_length)
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
    return encoder, decoder, input_lang, output_lang, max_length


if __name__ == "__main__":
    save_dir = os.path.join(dir_path, "model", "15.0_checkpoint.tar")
    encoder, decoder, input_lang, output_lang, max_length = load_model(save_dir)
    test_sentence = input()
    decoded_words, _ = evaluate(input_lang, output_lang, encoder, decoder, test_sentence, max_length)
    print(''.join(s for s in decoded_words[:-1]))