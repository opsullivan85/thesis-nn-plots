import sys

sys.path.append("../PlotNeuralNet/")
from pycore.tikzeng import *

# defined your arch
plot = [
    to_head(".."),
    to_cor(),
    to_begin(),
]

linear_scale = 0.5

z = 0
plot.append(
    to_Conv(
        "input",
        18,
        "",
        offset=f"({z},0,0)",
        height=1,
        depth=18*linear_scale,
        width=1,
        caption="Input",
    ),
)

prev_layer = "input"

z += 2
plot.append(
    to_Conv(
        "linear1",
        64,
        "",
        offset=f"({z},0,0)",
        height=1,
        depth=64*linear_scale,
        width=1,
        caption="Linear",
    )
)
plot.append(to_connection(prev_layer, "linear1"))
prev_layer = "linear1"

z += 2
plot.append(
    to_Conv(
        "linear2",
        64,
        "",
        offset=f"({z},0,0)",
        height=1,
        depth=64*linear_scale,
        width=1,
        caption="Linear",
    )
)
plot.append(to_connection(prev_layer, "linear2"))
prev_layer = "linear2"

z += 2
plot.append(
    to_Conv(
        "conv",
        8,
        2,
        offset=f"({z},0,0)",
        height=8,
        depth=8,
        width=1,
        caption="Conv2d",
    )
)
plot.append(to_connection(prev_layer, "conv"))
prev_layer = "conv"

# z += 3.5
# plot.append(
#     to_Conv(
#         "linear3",
#         128,
#         "",
#         offset=f"({z},0,0)",
#         height=1,
#         depth=128*linear_scale,
#         width=1,
#         caption="Linear",
#     )
# )
# plot.append(to_connection(prev_layer, "linear3"))
# prev_layer = "linear3"

z += 2
plot.append(
    to_Conv(
        "output",
        5,
        5,
        offset=f"({z},0,0)",
        height=5,
        depth=5,
        width=1,
        caption="Footstep\\\\Preferences",
    )
)
plot.append(to_connection(prev_layer, "output"))
prev_layer = "output"

z += 2
plot.append(
    to_Conv(
        "temp",
        5,
        5,
        offset=f"({z},0,0)",
        height=5,
        depth=5,
        width=1,
        caption="",
    )
)
plot.append(to_connection(prev_layer, "temp"))
prev_layer = "temp"



plot.append(to_end())


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(plot, namefile + ".tex")


if __name__ == "__main__":
    main()
