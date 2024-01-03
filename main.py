import click
import torch
import matplotlib.pyplot as plt
import os
from model import MyAwesomeModel
from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")

    model = MyAwesomeModel()
    train_loader, _ = mnist()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 30

    train_losses = []
    for epoch in range(epochs):
        train_loss = 0
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)

            optimizer.zero_grad()
            y_pred = model(images)
            loss = criterion(y_pred, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch: {epoch}, Loss: {avg_train_loss}")  
    torch.save(model, "../../s1_development_environment/exercise_files/final_exercise/model.pt")

    # Plot the training loss
    plt.plot(range(epochs), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs. Epoch")
    plt.show()



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_loader = mnist()
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0
    with torch.no_grad():  
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            y_pred = model(images)
            loss = criterion(y_pred, labels)

            ps = torch.exp(y_pred)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

    print(f"Accuracy: {accuracy.item() / len(test_loader)}")

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
