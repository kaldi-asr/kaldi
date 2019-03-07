#!/usr/bin/env python

# Copyright 2017    Joachim Fainberg.

# This script converts nnet2 models into nnet3 models.
# It requires knowledge of valid components which
# can be modified in the configuration section below.

from __future__ import print_function
import argparse, os, tempfile, logging, sys, shutil, fileinput, re
from collections import defaultdict, namedtuple
import numpy as np
sys.path.insert(0, 'steps/')
import libs.nnet3.train.common as common_train_lib
import libs.common as common_lib

# Begin configuration section
# Components and their corresponding node names

NODE_NAMES = {
    "<AffineComponent>":"affine",
    "<AffineComponentPreconditioned>":"affine",
    "<AffineComponentPreconditionedOnline>":"affine",
    "<BlockAffineComponent>":"affine",
    "<BlockAffineComponentPreconditioned>":"affine",
    "<SigmoidComponent>":"nonlin",
    "<TanhComponent>":"nonlin",
    "<PowerComponent>":"nonlin",
    "<RectifiedLinearComponent>":"nonlin",
    "<SoftHingeComponent>":"nonlin",
    "<PnormComponent>":"nonlin",
    "<NormalizeComponent>":"renorm",
    "<MaxoutComponent>":"maxout",
    "<MaxpoolingComponent>":"maxpool",
    "<ScaleComponent>":"rescale",
    "<DropoutComponent>":"dropout",
    "<SoftmaxComponent>":"softmax",
    "<LogSoftmaxComponent>":"log-softmax",
    "<FixedScaleComponent>":"fixed-scale",
    "<FixedAffineComponent>":"fixed-affine",
    "<FixedLinearComponent>":"fixed-linear",
    "<FixedBiasComponent>":"fixed-bias",
    "<PermuteComponent>":"permute",
    "<AdditiveNoiseComponent>":"noise",
    "<Convolutional1dComponent>":"conv",
    "<SumGroupComponent>":"sum-group",
    "<DctComponent>":"dct",
    "<SpliceComponent>":"splice",
    "<SpliceMaxComponent>":"splice"
}

SPLICE_COMPONENTS = [c for c in NODE_NAMES if "Splice" in c]
AFFINE_COMPONENTS = [c for c in NODE_NAMES if "Affine" in c]

KNOWN_COMPONENTS = list(NODE_NAMES.keys())
# End configuration section

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(filename)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

def GetArgs():
    parser = argparse.ArgumentParser(
        description="Converts nnet2 into nnet3 models.",
        epilog="""e.g. steps/nnet3/convert_nnet2_to_nnet3.py 
                  exp/tri4_nnet2 exp/tri4_nnet3""")
    parser.add_argument("--tmpdir", type=str, default="./",
                        help="Custom location for the temporary directory.")
    parser.add_argument("--skip-cleanup", action='store_true',
                        help="Will not remove the temporary directory.")
    parser.add_argument("--model", type=str, default='final.mdl',
                        help="Choose a specific model to convert.")
    parser.add_argument("--binary", type=str, default="true", 
                        choices=["true","false"], 
                        help="Whether to write the model in binary or not.")
    parser.add_argument("nnet2_dir", metavar="src-nnet2-dir", type=str,
                        help="")
    parser.add_argument("nnet3_dir", metavar="src-nnet3-dir", type=str,
                        help="")

    print(' '.join(sys.argv))

    args = parser.parse_args()

    if not os.path.exists(args.nnet3_dir):
        os.makedirs(args.nnet3_dir)
    if args.tmpdir and not os.path.exists(args.tmpdir):
        os.makedirs(args.tmpdir)

    return args

class Nnet3Model(object):
    """Holds configuration for an Nnet3 model."""
    
    def __init__(self):
        self.input_dim = -1
        self.output_dim = -1
        self.counts = defaultdict(int)
        self.num_components = 0
        self.components_read = 0
        self.config = ""
        self.transition_model = ""
        self.priors = ""
        self.components = []

    def add_component(self, component, pairs):
        """Adds components to the model. 
        
        Takes a dictionary of key-value pairs.
        """
        self.components_read += 1

        Component = namedtuple("Component", "ident component pairs")

        if "<InputDim>" in pairs and self.input_dim == -1:
            self.input_dim = pairs["<InputDim>"]

        # remove nnet2 specific tokens and catch descriptors
        if component == "<PnormComponent>" and "<P>" in pairs:
            pairs.pop("<P>")
        elif component in SPLICE_COMPONENTS:
            self.components.append(Component("splice", component, pairs))
            return

        # format pairs: {'<InputDim>':43} -> {'input-dim':43}
        pairs = ["{0}={1}".format(token_to_string(key), pairs[key]) for key in pairs]
        
        # keep track of layer type number (e.g. affine3)
        node_name = NODE_NAMES[component]
        self.counts[node_name] += 1

        # e.g. affine3
        ident = node_name + str(self.counts[node_name])

        # <PnormComponent> -> PnormComponent
        component = component[1:-1]

        self.components.append(Component(ident, component, pairs))

    def write_config(self, filename):
        """Write config to filename."""
        logger.info("Writing config to {0}".format(filename))

        self.config = filename
        with open(filename, 'w') as f:
            for component in self.components:
                if component.ident == "splice":
                    continue
                config_string = ' '.join(component.pairs)

                f.write("component name={name} type={comp_type} {config_string}"
                        "\n".format(name=component.ident, 
                                    comp_type=component.component, 
                                    config_string=config_string))

            f.write("\n# Component nodes\n")
            f.write("input-node name=input dim={0}\n".format(self.input_dim))
            previous_component = "input"
            for component in self.components:
                if component.ident == "splice":
                    # Create splice string for the next node
                    previous_component = make_splice_string(previous_component, 
                                                   component.pairs["<Context>"])
                    continue
                f.write("component-node name={name} component={name} "
                        "input={inp}\n".format(name=component.ident, 
                                               inp=previous_component))
                previous_component = component.ident
            logger.warning("Assuming linear objective.")
            f.write("output-node name=output input={inp} objective={obj}"
                    "\n".format(inp=previous_component, obj='linear'))

    def write_model(self, model, binary="true"):
        if not os.path.exists(self.config):
            raise IOError("Config file {0} does not exist.".format(self.config))

        # write raw model
        common_lib.execute_command("nnet3-init --binary=true {0} {1}"
            .format(self.config, os.path.join(tmpdir, "nnet3.raw")))

        # add transition model
        common_lib.execute_command("nnet3-am-init --binary=true {0} {1} {2}"
            .format(self.transition_model, os.path.join(tmpdir, "nnet3.raw"),
                    os.path.join(tmpdir, "nnet3_no_prior.mdl")))

        # add priors
        common_lib.execute_command("nnet3-am-adjust-priors "
                                     "--binary={0} {1} {2} {3}"
            .format(binary, os.path.join(tmpdir, "nnet3_no_prior.mdl"), 
                    self.priors, model))

def parse_nnet2_to_nnet3(line_buffer):
    """Reads an Nnet2 model into an Nnet3 object.

    Parses by passing line_buffer objects depending upon the
    current place or component being read.

    Returns Nnet3 object.
    """
    model = Nnet3Model()

    # <TransitionModel> ...
    model.transition_model = parse_transition_model(line_buffer)
    
    # <Nnet> <NumComponents> ...
    line, model.num_components = parse_nnet2_header(line_buffer)

    # Parse remaining components
    while True:
        if line.startswith("</Components>"):
            break
        component, pairs = parse_component(line, line_buffer)
        model.add_component(component, pairs)
        line = next(line_buffer)

    model.priors = parse_priors(line, line_buffer)
    
    if model.components_read != model.num_components:
        logger.error("Did not read all components succesfully: {0}/{1}"
                     .format(model.components_read, model.num_components))

    return model

def parse_transition_model(line_buffer):
    """Writes transition model to text file.
    
    Returns filename.
    """
    line = next(line_buffer)
    assert line.startswith("<TransitionModel>")

    transition_model = os.path.join(tmpdir, "transition_model")

    with open(transition_model, 'w') as fc:
        fc.write(line)
        
        while True:
            line = next(line_buffer)
            fc.write(line)
            if line.startswith("</TransitionModel>"):
                break

        return transition_model

def parse_nnet2_header(line_buffer):
    """Returns number of components in Nnet2 header."""
    line = next(line_buffer)
    assert line.startswith("<Nnet>")

    line = consume_token("<Nnet>", line)
    num_components = int(line.split()[1])
    line = line.partition(str(num_components))[2]
    line = consume_token("<Components>", line)

    return line, num_components 
                
def parse_component(line, line_buffer):
    component = line.split()[0]
    pairs = {}

    if component in SPLICE_COMPONENTS:
        pairs = parse_splice_component(component, line, line_buffer)
    elif component in AFFINE_COMPONENTS:
        pairs = parse_affine_component(component, line, line_buffer)
    elif component == "<FixedScaleComponent>":
        pairs = parse_fixed_scale_component(component, line, line_buffer)
    elif component == "<FixedBiasComponent>":
        pairs = parse_fixed_bias_component(component, line, line_buffer)
    elif component == "<SumGroupComponent>":
        pairs = parse_sum_group_component(component, line, line_buffer)
    elif component in KNOWN_COMPONENTS:
        pairs = parse_standard_component(component, line, line_buffer)
    else:
        raise LookupError("Unrecognised component, {0}.".format(component))

    parse_end_of_component(component, line, line_buffer)

    return component, pairs

def parse_standard_component(component, line, line_buffer):
    # Ignores stats such as ValueSum and DerivSum
    line = consume_token(component, line)
    pairs = re.findall("(<\w+>) ([\w.]+)", line)

    return dict(pairs)

def parse_fixed_scale_component(component, line, line_buffer):
    line = consume_token(component, line)
    line = consume_token("<Scales>", line)

    scales = np.array([parse_vector(line)])

    _, filename = tempfile.mkstemp(dir=tmpdir)
    with open(filename, 'w') as f:
        f.write("[ ")
        np.savetxt(f, scales, newline='')
        f.write(" ]")

    return {"<Scales>" : filename}

def parse_sum_group_component(component, line, line_buffer):
    line = consume_token(component, line)
    line = consume_token("<Sizes>", line)

    sizes = line.strip().strip("[]").strip().replace(' ', ',')

    return {"<Sizes>" : sizes}

def parse_fixed_bias_component(component, line, line_buffer):
    line = consume_token(component, line)
    line = consume_token("<Bias>", line)

    scales = np.array([parse_vector(line)])

    _, filename = tempfile.mkstemp(dir=tmpdir)
    with open(filename, 'w') as f:
        f.write("[ ")
        np.savetxt(f, scales, newline='')
        f.write(" ]")

    return {"<Bias>" : filename}

def parse_splice_component(component, line, line_buffer):
    if component == "<SpliceMaxComponent>":
        raise NotImplementedError("Script doesn't support SpliceMaxComponent.")

    line = consume_token(component, line)
    line = consume_token("<InputDim>", line)
    [input_dim, _, line] = line.strip().partition(' ')
    line = consume_token("<Context>", line)
    context = line.strip()[1:-1].split()

    return {"<InputDim>" : input_dim, "<Context>" : context}

def parse_end_of_component(component, line, line_buffer):
    # Keeps reading until it hits the end tag for component
    end_component = "</" + component[1:]

    while end_component not in line:
        line = next(line_buffer)

    return

def parse_affine_component(component, line, line_buffer):
    assert ("<LinearParams>" in line)

    pairs = dict(re.findall("(<\w+>) ([\w.]+)", line))

    # read the linear params and bias and convert it to a matrix
    weights = parse_weights(line_buffer)
    bias = parse_bias(next(line_buffer))

    matrix = np.concatenate([weights, bias.T], axis=1)

    # write matrix and return pairs with filename
    _, filename = tempfile.mkstemp(dir=tmpdir)
    with open(filename, 'w') as f:
        f.write("[ ")
        np.savetxt(f, matrix)
        f.write(" ]")

    pairs["<Matrix>"] = filename

    return pairs

def parse_weights(line_buffer):
    weights = []

    while True:
        line = next(line_buffer)

        if line.strip().endswith("["):
            continue
        elif line.strip().endswith("]"):
            weights.append(parse_vector(line))
            break
        else:
            weights.append(parse_vector(line))

    return np.array(weights)

def parse_bias(line):
    if "<BiasParams>" in line:
        line = consume_token("<BiasParams>", line)

    return np.array([parse_vector(line)])

def parse_vector(line):
    vector = line.strip().strip("[]")
    return np.array([float(x) for x in vector.split()], dtype="float32")

def parse_priors(line, line_buffer):
    vector = parse_vector(line.partition('[')[2])
    priors = os.path.join(tmpdir, "priors")

    with open(priors, 'w') as f:
        f.write("[ ")
        np.savetxt(f, vector, newline=' ')
        f.write(" ]")

    return priors

def token_to_string(token):
    """Converts tokens to lowercase, hyphen-bounded strings.

    E.g. <InputDim> -> input-dim
    """
    string = token[1:-1]
    string = re.sub(r"((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))", r'-\1', string).lower()
    return string

def consume_token(token, line):
    """Returns line without token"""
    if token != line.split(None, 1)[0]:
        logger.error("Unexpected token, expected '{0}', got '{1}'."
              .format(token, line.split(None, 1)[0]))

    return line.partition(token)[2]

def make_splice_string(nodename, context):
    """Generates splice string from a list of context.

    E.g. make_splice_string("renorm4", [-4, 4])
    returns "Append(Offset(renorm4, -4), Offset(renorm4, 4))"
    """
    assert type(context) == list, "context argument must be a list"
    string = ["Offset({0}, {1})".format(nodename, i) for i in context]
    string = "Append(" + ", ".join(string) + ")"
    return string

tmpdir = ""

def Main():
    args = GetArgs()
    logger.info("Converting nnet2 model {0} to nnet3 model {1}"
                .format(os.path.join(args.nnet2_dir, args.model), 
                        os.path.join(args.nnet3_dir, args.model)))
    global tmpdir
    tmpdir = tempfile.mkdtemp(dir=args.tmpdir) 

    # Convert nnet2 model to text and remove preconditioning
    common_lib.execute_command("nnet-am-copy "
            "--remove-preconditioning=true --binary=false {0}/{1} {2}/{1}"
            .format(args.nnet2_dir, args.model, tmpdir))

    # Parse nnet2 and return nnet3 object
    with open(os.path.join(tmpdir, args.model)) as f:
        nnet3 = parse_nnet2_to_nnet3(f)

    # Write model
    nnet3.write_config(os.path.join(tmpdir, "config"))
    nnet3.write_model(os.path.join(args.nnet3_dir, args.model), 
                      binary=args.binary)
        
    if not args.skip_cleanup:
        shutil.rmtree(tmpdir)
    else:
        logger.info("Not removing temporary directory {0}".format(tmpdir))
     
    logger.info("Wrote nnet3 model to {0}".format(os.path.join(args.nnet3_dir, 
                                                  args.model)))

if __name__ == "__main__":
    Main()
