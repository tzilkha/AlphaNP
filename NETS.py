import sys
import numpy as np
import random

from TSP import TSPGame

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *     
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

import torch
from torch import nn
import torch.nn.functional as F

import torch_geometric
from torch_geometric.nn import GCNConv, global_mean_pool, ARMAConv, XConv, SAGEConv
from torch_geometric.data import Data, DataLoader

class ConvolutionalNN():
    def __init__(self, args):
        self.action_size = args['num_node']
        self.args = args
        self.create_net()
        self.HISTORY = False

    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        print("Training...")
        input_boards, input_graphs, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray([boards[-1] for boards in input_boards])
        input_graphs = np.asarray(input_graphs)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        
        boards_graphs_list = [(input_graphs[i], input_boards[i]) for i in range(0, len(input_boards))] 
        input = [np.concatenate([*i], axis=1) for i in boards_graphs_list]
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.args['patience'])
        
        hist = self.model.fit(x = [input], y = [target_pis, target_vs], validation_split=self.args['validation_split'],
                       batch_size = self.args.batch_size, epochs = self.args.epochs, verbose=1, callbacks=[es])
        return hist

    def predict(self, board, graph):
        """
        board: np array with board
        """        
        merged = np.concatenate([graph, board], axis=1)
        merged = merged[np.newaxis, :, :]
        
        # run
        pi, v = self.model.predict(merged)
        return pi[0], v[0]
    
    def create_net(self):
        # Neural Net
        self.input_boards = Input(shape=(self.action_size, 4))   

        x_image = Reshape((self.action_size, 4, 1))(self.input_boards)           
        h_conv1 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 2, padding='same')(x_image)))     
        h_conv4 = Activation('relu')(BatchNormalization(axis=3)(Conv2D(self.args.num_channels, 2, padding='valid')(h_conv1)))     
        h_conv4_flat = Flatten()(h_conv4)       
        s_fc1 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(1024)(h_conv4_flat)))) 
        s_fc2 = Dropout(self.args.dropout)(Activation('relu')(BatchNormalization(axis=1)(Dense(512)(s_fc1))))       
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(s_fc2)  
        self.v = Dense(1, activation='relu', name='v')(s_fc2)                  

        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.args.lr))
         
    def save_model(self, filename):
        model_json = self.model.to_json()
        with open('saved_models/' + filename + ".json", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        self.model.save_weights('saved_models/' + filename + ".h5")
        print("Saved model to disk")
        
    def load_model(self, filename):
        # load json and create model
        json_file = open('saved_models/' + filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights('saved_models/' + filename + ".h5")
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.args.lr))  
        print("Loaded model from disk")
        


class RecurrentNN():
    def __init__(self, args):
        # game params
        self.action_size = args['num_node']
        self.num_node = args['num_node']
        self.board_size = (args['num_node'],2)
        self.args = args
        self.create_net()
        # Conv no history
        self.HISTORY = True       
        
    def create_net(self):
        self.input_boards = Input(shape=(self.num_node, self.num_node*4))
        lstm = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(self.input_boards)
        dense = Dense(128, activation='sigmoid')(lstm)
        self.pi = Dense(self.action_size, activation='softmax', name='pi')(dense)
        self.v = Dense(1, activation='relu', name='v')(dense)
        
        self.model = Model(inputs=self.input_boards, outputs=[self.pi, self.v])
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.args.lr))        

    def predict(self, board_list, graph):
        input = self.prepare_input(board_list,graph)
        pi, v = self.model.predict(input)


        #print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return pi[0], v[0]
    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        print("Training...")
        input_boards, input_graphs, target_pis, target_vs = list(zip(*examples))
        input_boards = np.asarray(input_boards)
        input_graphs = np.asarray(input_graphs)
        target_pis = np.asarray(target_pis)
        target_vs = np.asarray(target_vs)
        boards_graphs_list = [(input_boards[i], input_graphs[i]) for i in range(0, len(input_boards))] 
        prepared_inputs = np.asarray([self.prepare_input(*i)[0] for i in boards_graphs_list])
        
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=self.args['patience'])
        
        hist = self.model.fit(x = [prepared_inputs], y = [target_pis, target_vs], validation_split=self.args['validation_split'],
                       batch_size = self.args.batch_size, epochs = self.args.epochs, verbose=1, callbacks=[es])
        return hist

    #
    # State representation is a history of states with the graph appended to them and the whole thing shaped to array.
    #
    #      [graph, visited, camefrom]  t0
    #      [graph, visited, camefrom]  t1
    #      [graph, visited, camefrom]  t2
    #      ..
    #      ..
    #      [graph, visited, camefrom]  tn
    #
    # Where tn is the current state and any timestamp that came before the first move is as such
    #      
    #      [graph,0,0]
    #
    
    def prepare_input(self,board_list, graph):
        if self.args['history_length'] is None:
            self.args['history_length'] = self.num_node
        while len(board_list)<self.args['history_length']:
            board_list = [np.zeros(self.board_size)] + board_list
        board_list = [x.transpose() for x in board_list]
        input = np.array(board_list).reshape(len(board_list), self.num_node*2)
        [graph]*self.num_node
        graph_data = np.array(([np.array(graph).reshape(self.num_node*2)]*len(board_list))).reshape(len(board_list), self.num_node*2)
        input = np.stack((np.array(graph_data), np.array(input)), axis=1).reshape(1,self.num_node,self.num_node*4)
        return(input)
    
    def save_model(self, filename):
        model_json = self.model.to_json()
        with open('saved_models/' + filename + ".json", "w") as json_file:
            json_file.write(model_json)
            
        # serialize weights to HDF5
        self.model.save_weights('saved_models/' + filename + ".h5")
        print("Saved model to disk")
        
    def load_model(self, filename):
        # load json and create model
        json_file = open('saved_models/' + filename + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
        # load weights into new model
        self.model.load_weights('saved_models/' + filename + ".h5")
        self.model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(self.args.lr))  
        print("Loaded model from disk")
        
     
    
class GraphConvolutionalNN():
    def __init__(self, args):
        
        self.args = args
        self.model = gnn()

        self.optimizer = torch.optim.Adam(params = self.model.parameters(),lr=self.args['lr'])
        self.loss_func = nn.MSELoss()
        
        self.HISTORY = False
        
    def create_net(self):
        return
    
    def train(self, examples):
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        losses = {'loss':[], 'v_loss':[], 'val_v_loss':[], 'val_pi_loss':[], 'pi_loss':[], 'val_loss':[]}
        self.model.train()
        
        count_improvement = 0
        best_last = np.inf
        
        examples = self.convert_dataset(examples)
        
        for epoch in range(self.args['epochs']):

            epoch_losses = {'loss':0, 'v_loss':0, 'val_v_loss':0, 'val_pi_loss':0, 'pi_loss':0, 'val_loss':0}
            
            random.shuffle(examples)

            cut = int(len(examples)*self.args['validation_split'])
            testset = examples[:cut]
            trainset = examples[cut:]
            
            testset_n = len(testset)
            trainset_n = len(testset)
            
            totloss_v = 0
            totloss_p = 0
                        
            if (epoch+1)%10 == 0: print("Epoch", str(epoch+1)+"/"+str(self.args['epochs']), "Loss -", losses['loss'][-1])
                
            for example in trainset:
                g,p,v = example
                
                pred_choices, pred_value = self.model(g, g.y)
                
                p = [[i] for i in p]
                
                loss_p = self.loss_func(pred_choices.double(), torch.tensor(p).double())

                loss_v =  self.loss_func(pred_value.double(), torch.tensor(v).double())
                        
                loss = loss_p + loss_v
                
                epoch_losses['pi_loss'] += loss_p.item()/trainset_n
                epoch_losses['v_loss'] += loss_v.item()/trainset_n
                epoch_losses['loss'] += loss.item()/trainset_n
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            for example in testset:
                g,p,v = example
                
                pred_choices, pred_value = self.model.forward(g, g.y)
                
                loss_p = self.loss_func(pred_choices.double(), torch.tensor(p).double())
                
                loss_v = self.loss_func(pred_value.double(), torch.tensor(v).double())
                
                loss = loss_p.item() + loss_v.item()
                
                epoch_losses['val_pi_loss'] += loss_p.item()/testset_n
                epoch_losses['val_v_loss'] += loss_v.item()/testset_n
                epoch_losses['val_loss'] += loss/testset_n
                
            for key in losses.keys():
                losses[key] += [epoch_losses[key]]
                                                         
            # Check patience and early stopping
            if best_last > losses['loss'][-1]:
                count_improvement = 0
                best_last = losses['loss'][-1]
            else:
                count_improvement += 1
                if count_improvement >= self.args['patience']:
                    print("No improvement - Early Stopping")
                    break
                else:
                    count_improvement += 1
    
        return losses
    
    def convert_dataset(self, examples):
        graph_examples = []
        for example in examples:
            g,p,v = self.to_graph(example)
            graph_examples += [(g, p, v)]

        return graph_examples
        
    def to_graph(self, s):
        hist, graph, p, v = s

        cur = hist[-1]

        graph_ = torch.tensor(graph).to(dtype=torch.float)

        current_v = [i for i,j in enumerate(hist[-1]) if j[1]==1][0]
        available_v = [i for i,j in enumerate(hist[-1]) if j[0]==0]

        choices = torch.zeros(self.args['num_node'], dtype=torch.bool)
        choices[available_v] = 1

        edges = torch.zeros((2, len(available_v)), dtype=torch.long)
        for i,j in enumerate(available_v):
            edges[0, i] = current_v
            edges[1, i] = j

        g = Data(pos=graph_, edge_index=edges, y=choices)
    
        return g, p, v
    
    def predict(self, hist, graph):
        """
        board: np array with board
        """        
        graph_ = torch.tensor(graph).to(dtype=torch.float)
            
        if self.HISTORY:
            current_v = [i for i,j in enumerate(hist[-1]) if j[1]==1][0]
            available_v = [i for i,j in enumerate(hist[-1]) if j[0]==0]
        else:
            current_v = [i for i,j in enumerate(hist) if j[1]==1][0]
            available_v = [i for i,j in enumerate(hist) if j[0]==0]
        
        choices = torch.zeros(self.args['num_node'], dtype=torch.bool)
        choices[available_v] = 1

        edges = torch.zeros((2, len(available_v)), dtype=torch.long)
        for i,j in enumerate(available_v):
            edges[0, i] = current_v
            edges[1, i] = j

        g = Data(pos=graph_, edge_index=edges, y=choices)

        p, v = self.model.forward(g, choices)
        return np.array([i[0] for i in np.array(p.detach())]), np.array(v.detach())[0][0]
         
    def save_model(self, filename):
        torch.save(self.model.state_dict(), './saved_models/'+filename)
        return 
    
    def load_model(self, filename):
        self.model.load_state_dict(torch.load('./saved_models/'+filename))
        print('Loaded model from disk')
        return




class gnn(nn.Module):
    def __init__(self, d=2):
        self.d = d
        super(gnn, self).__init__()
        self.conv1 = GCNConv(d,  32)
        self.conv2 = GCNConv(32, 16)
        self.conv3 = GCNConv(16, 1)
        self.fc    = nn.Linear(16, 1)
    
    def forward(self, graph, choices):
        x, edges = graph.pos, graph.edge_index
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = self.conv2(x, edges)
        x = F.relu(x)
        
        c = self.conv3(x, edges)
        choice = F.softmax(c, dim=0)
        
        v = global_mean_pool(x, torch.zeros(graph.num_nodes, dtype=torch.long))
        value = self.fc(v)

        return choice, value
    

    
# build training dataset
def create_training_set(args, n_games = 100):
    data_set = []
    for i in range(n_games):
        t = TSPGame(args)
        train_v, train_pi = t.create_sample()
        for i in range(1,len(train_v)):
            boards = [j[0] for j in train_v[:i]]
            graph = t.graph
            data_set+=[(boards, graph, train_pi[i-1][1], train_v[i-1][1])]
    return data_set
            