from pykeen.datasets import PathDataSet, TriplesFactory, DataSet

class InductiveDataset(DataSet):

    def __init__(self,
                 transductive: str,
                 inductive: str,
                 create_inverse_triples: bool = True,):

        self.cache_root = "./data/"

        self.transductive_part = TriplesFactory(
            path=self.cache_root + transductive + "/train.txt",
            create_inverse_triples=create_inverse_triples
        )
        self.inductive_part = inductive

        self.inductive_inference = TriplesFactory(
            path=self.cache_root + inductive + "/train.txt",
            relation_to_id=self.transductive_part.relation_to_id,
            create_inverse_triples=create_inverse_triples
        )

        self.inductive_val = TriplesFactory(
            path=self.cache_root + inductive + "/valid.txt",
            entity_to_id=self.inductive_inference.entity_to_id,
            relation_to_id=self.transductive_part.relation_to_id
        )

        self.inductive_test = TriplesFactory(
            path=self.cache_root + inductive + "/test.txt",
            entity_to_id=self.inductive_inference.entity_to_id,
            relation_to_id=self.transductive_part.relation_to_id
        )


class Ind_FB15k237(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True,):

        super().__init__(transductive=f"fb237_v{version}",
                         inductive=f"fb237_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)


class Ind_WN18RR(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True, ):
        super().__init__(transductive=f"WN18RR_v{version}",
                         inductive=f"WN18RR_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)


class Ind_NELL(InductiveDataset):

    def __init__(self,
                 version: int = 1,
                 create_inverse_triples: bool = True, ):
        super().__init__(transductive=f"nell_v{version}",
                         inductive=f"nell_v{version}_ind",
                         create_inverse_triples=create_inverse_triples)

