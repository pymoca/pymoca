from pymoca import parser, tree

txt = """
package TreeModel
    package TreeTypes
        type Cell = Integer;
    end TreeTypes;
    package TreeParts
        type Oak = Real(nominal=1.0);
        model Trunk
            replaceable type Wood = Real;
            Wood t;
        end Trunk;
        constant Boolean e;
        model Branch
            extends Trunk(redeclare type Wood = Oak);
        end Branch;
        model Leaf
            TreeTypes.Cell c=2;
        end Leaf;
    end TreeParts;
    model Tree
        Wood w;
        TreeParts.Branch b;
        TreeParts.Leaf l(c=1);
        extends TreeParts.Trunk;
    end Tree;
end TreeModel;
"""
ast_tree = parser.parse(txt)
instance_tree = tree.InstanceTree(ast_tree)

instance = instance_tree.instantiate("TreeModel.Tree")
