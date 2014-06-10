from parsimonious.nodes import NodeVisitor


class ModelicaPrinter(NodeVisitor):

    def __init__(self):
        print

    def info(self, node):
        return '\033[94m {:s} \033[0m {:s}'.format(
            node.expr_name, node.full_text[node.start:node.end])

    def generic_visit(self, node, visited_children):
        print self.info(node)

    def visit_class_definition(self, node, visitied_children):
        print self.info(node)

    def visit_(self, node, visitied_children):
        pass

    def visit__(self, node, visitied_children):
        pass
