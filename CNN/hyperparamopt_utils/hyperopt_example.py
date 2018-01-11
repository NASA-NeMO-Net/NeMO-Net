from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys

X = []
y = []
X_val = []
y_val = []

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,1024), 
                    'dropout3': hp.uniform('dropout3', .25,.75)}
                    ]),

            'units1': hp.uniform('units1', 64,1024),
            'units2': hp.uniform('units2', 64,1024),

            'dropout1': hp.uniform('dropout1', .25,.75),
            'dropout2': hp.uniform('dropout2',  .25,.75),

            'batch_size' : hp.uniform('batch_size', 28,128),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(output_dim=params['units1'], input_dim = X.shape[1])) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout1']))

    model.add(Dense(output_dim=params['units2'], init = "glorot_uniform")) 
    model.add(Activation(params['activation']))
    model.add(Dropout(params['dropout2']))

    if params['choice']['layers']== 'three':
        model.add(Dense(output_dim=params['choice']['units3'], init = "glorot_uniform")) 
        model.add(Activation(params['activation']))
        model.add(Dropout(params['choice']['dropout3']))    

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'])

    model.fit(X, y, nb_epoch=params['nb_epochs'], batch_size=params['batch_size'], verbose = 0)

    pred_auc =model.predict_proba(X_val, batch_size = 128, verbose = 0)
    acc = roc_auc_score(y_val, pred_auc)
    print('AUC:', acc)
    sys.stdout.flush() 
    return {'loss': -acc, 'status': STATUS_OK}


trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print 'best: '
print best



## with hyperopt need to define space:
def space():
    space = {'num_layer' : hp.choice('num_layer',[{'layers':'add1'},{'layers':'add2'},
    {'layers':'add3'},{'layers':'add4'}]),

         'activation' : hp.choice('activation',['ELU(alpha=1.0)','Activation(tanh)']),
         'optimizer' : hp.choice('optimizer',['SGD(lr=0.03, decay=1e-7, momentum=0.15, nesterov=True)','RMSprop','Adadelta','Adam']),
         'dropout1' : hp.uniform('dropout1',0.25,0.75),
         'dropout2' : hp.uniform('dropout2',0.05, 0.5),
         'nb_epochs' :  150,
         #'units' : hp.quniform('units', 800,1400,2),
         'units' : hp.choice('units', [1024,1512,2048,2560]),
         'regularizer' : hp.choice('regularizer',['l2','activity_l2']),           
         }

# and then run the model with space:
def model(space,X_train,Y_train,X_test,Y_test):
    model = Sequential()
    model.add(Dense(output_dim=space['units'], input_dim=X_train.shape[1], init='he_uniform', W_regularizer=l2(l=0.0001)))
    print('it is ok add layer')