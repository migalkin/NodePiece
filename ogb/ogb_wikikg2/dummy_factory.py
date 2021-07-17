class DummyTripleFactory:

    '''
    This is to mimic pykeen's triple factory which is a default input for the KG Tokenizer
    '''

    def __init__(self, triples, ne, nr):

        self.mapped_triples = triples
        self.num_entities = ne
        self.num_relations = nr

        self.relation_to_id = {i: i for i in range(nr)}