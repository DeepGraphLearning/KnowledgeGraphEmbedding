FB15K-237 Knowledge Base Completion Dataset

This dataset contains knowledge base relation triples and textual mentions of Freebase entity pairs, as used in the work published in [1] and [2]. 
The knowledge base triples are a subset of the FB15K set [3], originally derived from Freebase. The textual mentions are derived from 200 million sentences from the ClueWeb12 [5] corpus coupled with Freebase entity mention annotations [4].


FILE FORMAT DETAILS

The files train.txt, valid.txt, and test.text contain the training, development, and test set knowledge base triples used in both [1] and [2].
The file text_cvsc.txt contains the textual triples used in [2] and the file text_emnlp.txt contains the textual triples used in [1].

The knowledge base triples contain lines like this:

/m/0grwj	/people/person/profession	/m/05sxg2

The format is:

mid1	relation	mid2

The separator is a tab character; the mids are Freebase ids of entities, and the relation is a single or a two-hop relation from Freebase, where an intermediate complex value type entity has been collapsed out.

The textual mentions files have lines like this:

/m/02qkt        [XXX]:<-nn>:fact:<-pobj>:in:<-prep>:game:<-nsubj>:'s:<ccomp>:pivot:<nsubj>:[YYY]    /m/05sb1    3

This indicates the mids of two Freebase entities, together with a fully lexicalized dependency path between the entities. The last element in the tuple is the number of occurrences of the specified entity pair with the given dependency path in sentences from ClueWeb12.
The dependency paths are specified as sequences of words (like the word "fact" above) and labeled dependency links (like <nsubj> above). The direction of traversal of a dependency arc is indicated by whether there is a - sign in front of the arc label "e.g." <-nsubj> vs <nsubj>.


REFERENCES

[1] Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. Representing text for joint embedding of text and knowledge bases.  In Proceedings of EMNLP 2015.
[2] Kristina Toutanova and Danqi Chen. Observed versus latent features for knowledge base and text inference. In Proceedings of the 3rd Workshop on Continuous Vector Space Models and Their Compositionality 2015.
[3] Antoine Bordes, Nicolas Usunier, Alberto Garcia Duran, Jason Weston, and Oksana Yakhnenko.  Translating embeddings for modeling multirelational data. In Advances in Neural Information Processing Systems (NIPS) 2013.
[4] Evgeniy Gabrilovich, Michael Ringgaard, and Amarnag Subramanya. FACC1: Freebase annotation of ClueWeb corpora, Version 1 (release date 2013-06-26, format version 1, correction level 0). http://lemurproject.org/clueweb12/FACC1/
[5] http://lemurproject.org/clueweb12/


CONTACT

Please contact Kristina Toutanova kristout@microsoft.com if you have questions about the dataset.
