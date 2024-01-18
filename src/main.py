from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

from ptdec.dec import DEC
from ptdec.model import train, predict
import ptsdae.model as ae
from ptdec.utils import cluster_accuracy

from autoencoder import DAE, init_weights
from mnist import CachedMNIST

def main():
    """
    Função adaptada do repositório abaixo para usar meu modelo de autoencoder
    na fase de inicialização e o Dec no restante.
    https://github.com/vlukiyanov/pt-dec
    """
    cuda=False
    batch_size=256
    finetune_epochs=100
    testing_mode=False
   
    #Aqui faço a leitura dos dados
    ds_train = CachedMNIST(train=True, cuda=cuda, testing_mode=testing_mode) 
    ds_val = CachedMNIST(train=False, cuda=cuda, testing_mode=testing_mode) 

    #------------------Parte do autoencoder----------------------
    autoencoder = DAE()
    autoencoder.apply(init_weights)

    print("Fase de treinamento.")
    ae_optimizer = SGD(params=autoencoder.parameters(), lr=0.1, momentum=0.9)
    ae.train(
        ds_train,
        autoencoder,
        cuda=cuda,
        validation=ds_val,
        epochs=finetune_epochs,
        batch_size=batch_size,
        optimizer=ae_optimizer,
        scheduler=StepLR(ae_optimizer, 100, gamma=0.1),
        corruption=0.2,
    )

    #------------------------------------------------------------------------------
    print("Fase do DEC.")
    model = DEC(cluster_number=10, hidden_dimension=10, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
        
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
        dataset=ds_train,
        model=model,
        epochs=100,
        batch_size=256,
        optimizer=dec_optimizer,
        stopping_delta=0.000001,
        cuda=cuda,
    )
    predicted, actual = predict(
        ds_train, model, 1024, silent=True, return_actual=True, cuda=cuda
    )
    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    _, accuracy = cluster_accuracy(actual, predicted)
    print("Acurácia final do DEC: %s" % accuracy)

if __name__ == "__main__":
    main()
