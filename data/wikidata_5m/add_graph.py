import os
import json

path = os.path.join(os.getcwd(), 'KnowledgeGraphEmbedding\data\wikidata_5m')
print(path)

class Graph():
    def __init__(self, graph):
        self.graph = graph
    
    def getGraph(self):
        return self.graph

    def getId2Entity(self):
        return self.graph['id2entity']
    
    def getEntity2Id(self):
        return self.graph['entity2id']

    def getId2Relation(self):
        return self.graph['id2relation']
    
    def getRelation2Id(self):
        return self.graph['relation2id']

def readId2Entity(graph: Graph, path=path):
    id_entity = graph.getId2Entity()
    print(len(id_entity))

    entity_file = open(os.path.join(path, 'entities.dict'), 'a+')
    # counter = 10
    i = 0
    for entity in id_entity:
        entity_file.write(f'{i}\t{entity}\n')
        i = i+1

def readEntity2Id(graph: Graph, path=path):
    entity_id = graph.getEntity2Id()
    print(type(entity_id))

def readId2Relation(graph: Graph, path=path):
    id_relation = graph.getId2Relation()
    print(len(id_relation))

    relation_file = open(os.path.join(path, 'relations.dict'), 'a+')
    # counter = 10
    i = 0
    for relation in id_relation:
        relation_file.write(f'{i}\t{relation}\n')
        i = i+1

def readRelation2Id(graph: Graph, path=path):
    relation_id = graph.getRelation2Id()
    print(type(relation_id))

def run(path):
    graph_path = os.path.join(path, 'graph_id.json')
    graph_file = open(graph_path, 'r')
    graph: Graph = Graph(json.load(graph_file))

    print(graph.getGraph().keys())
    readId2Entity(graph, path)
    # readEntity2Id(graph, path)
    readId2Relation(graph, path)
    # readRelation2Id(graph, path)

if __name__=='__main__':
    run(path=path)