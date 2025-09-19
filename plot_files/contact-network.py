import sys

sys.path.append("../PlotNeuralNet/")
from pycore.tikzeng import *

# defined your arch
plot = [
    to_head(".."),
    to_cor(),
    to_begin(),
]

z = 0
plot.append(
    to_Conv(
        "state_input",
        43,
        "",  # type: ignore
        offset=f"({z},0,0)",
        height=1,
        depth=43,
        width=1,
        caption="State\\\\Input",
    ),
)

prev_layer = "state_input"
denses = [64, 64, 64]
for i, size in enumerate(denses):
    z += 2
    plot.append(
        to_Conv(
            f"state_dense{i+1}",
            size,
            "",  # type: ignore
            offset=f"({z},0,0)",
            height=1,
            depth=size,
            width=1,
            caption="Dense",
        )
    )
    plot.append(to_connection(prev_layer, f"state_dense{i+1}"))
    prev_layer = f"state_dense{i+1}"

z += 2
plot.append(
    to_Conv(
        f"output",
        5,
        5,
        offset=f"({z},0,0)",
        height=5,
        depth=5,
        width=1,
        caption="Output",
    )
)
plot.append(to_connection(prev_layer, f"output"))

z += 2
plot.append(
    to_Conv(
        f"placeholder",
        5,
        5,
        offset=f"({z},0,0)",
        height=5,
        depth=5,
        width=1,
        caption="",
    )
)
plot.append(to_connection("output", f"placeholder"))

plot.append(to_end())


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(plot, namefile + ".tex")


if __name__ == "__main__":
    main()
