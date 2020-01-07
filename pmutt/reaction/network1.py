import networkx as nx


def create_reaction_path(reactions, reactants, products, reservoir=None):
    """Create a reaction network

    Parameters
    ----------

    """
    # Create graph by adding reactants and products
    g = nx.Graph()
    g.add_node(reactants)
    g.add_node(products)
    # Compile requirements for reactions. Negative values indicate reverse
    # direction
    reactions_req = {}
    for i, reaction in enumerate(reactions):
        reactions_req[i] = {species.name: stoich for species, stoich in \
                            zip(reaction.reactants, reaction.reactants_stoich)}
        reactions_req[-i] = {species.name: stoich for species, stoich in \
                            zip(reaction.products, reaction.products_stoich)}
    # Identify possible reactions for each node
    for node in g.nodes:
        pass



