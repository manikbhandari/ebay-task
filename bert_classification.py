import os
import torch
import torch.optim as optim
import logging
import argparse

from transformers import BertTokenizer
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
from sklearn.metrics import accuracy_score
from models import BERT
from utils import save_checkpoint, load_checkpoint, init_logger, logger


def train(model,
          optimizer,
          train_loader,
          valid_loader,
          num_epochs,
          eval_every,
          ckpt_dir,
          best_valid_loss=float("Inf"),
          device='cpu'):
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    model.train()
    for epoch in range(num_epochs):
        for (id, text, label), _ in train_loader:
            print(f"Training step: {global_step}", end='\r')
            label = label.type(torch.LongTensor)
            label = label.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, label)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (id, text, label), _ in valid_loader:
                        label = label.type(torch.LongTensor)
                        label = label.to(device)
                        text = text.type(torch.LongTensor)
                        text = text.to(device)
                        output = model(text, label)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                logger.info('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                            .format(epoch + 1, num_epochs, global_step, num_epochs * len(train_loader),
                                    average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint(os.path.join(ckpt_dir, 'best_model.pt'), model, best_valid_loss)

    logger.info('Finished Training!')


def evaluate(model, test_loader, device='cpu'):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (id, text, label), _ in test_loader:
            label = label.type(torch.LongTensor)
            label = label.to(device)
            text = text.type(torch.LongTensor)
            text = text.to(device)
            output = model(text, label)

            _, output = output
            y_pred.extend(torch.argmax(output, 1).tolist())
            y_true.extend(label.tolist())

    logger.info(f'Accuracy: {accuracy_score(y_true, y_pred)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', help="log file")
    parser.add_argument('--bert_type', default='bert-base-uncased', help="type of bert model")
    parser.add_argument('--ckpt_dir', default='results/', help="Directory for saving the checkpoints.")
    parser.add_argument('--train_file', default='train.csv',
                        help="csv file for training, must be in the format id, text, label")
    parser.add_argument('--val_file', default='valid.csv',
                        help="csv file for validation, must be in the format id, text, label")
    parser.add_argument('--test_file', default='test.csv',
                        help="csv file for testing, must be in the format id, text, label")
    parser.add_argument('--device', default='cuda', help="Use cpu for training on cpu and cuda for GPU")
    parser.add_argument('--batch_size', type=int, default=8, help="Adjust batch size according to available memory")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs used for training")
    parser.add_argument('--eval_every', type=int, default=1000, help="Evaluate after these many steps")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    init_logger(args.log_file, logging.INFO)
    tokenizer = BertTokenizer.from_pretrained(args.bert_type)

    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    id_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False,
                       batch_first=True, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
    fields = [('id', id_field), ('text', text_field), ('label', label_field)]

    train_dataset, valid_dataset, test_dataset = TabularDataset.splits(
        path='.', train=args.train_file, validation=args.val_file,
        test=args.test_file, format='CSV', fields=fields, skip_header=True)

    train_iter = BucketIterator(train_dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                device=args.device, train=True, sort=True, sort_within_batch=True)
    valid_iter = BucketIterator(valid_dataset, batch_size=args.batch_size, sort_key=lambda x: len(x.text),
                                device=args.device, train=True, sort=True, sort_within_batch=True)
    test_iter = Iterator(test_dataset, batch_size=args.batch_size,
                         device=args.device, train=False, shuffle=False, sort=False)

    logger.info("Initializations done")

    model = BERT().to(args.device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-6)

    logger.info("Begin Training")
    train(model=model, optimizer=optimizer, device=args.device, train_loader=train_iter, valid_loader=valid_iter,
          num_epochs=args.epochs, eval_every=args.eval_every, ckpt_dir=args.ckpt_dir)
    print("Training over")

    best_model = BERT().to(args.device)
    load_checkpoint(os.path.join(args.ckpt_dir, 'best_model.pt'), best_model, device=args.device)
    evaluate(best_model, test_iter, device=args.device)
