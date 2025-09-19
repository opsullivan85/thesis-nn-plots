import sys

sys.path.append("../PlotNeuralNet/")
from pycore.tikzeng import *

offset = 1
cords = [
    (-offset, offset),
    (offset, offset),
    (offset, -offset),
    (-offset, -offset),
]

legs = []
for i, (x, y) in enumerate(cords):

    z = 0
    label = r"Footstep\\Preferences" if i == 0 else ""
    legs.append(
        to_Conv(
            f"input_{i}",
            5,
            5,
            offset=f"({z}, {x}, {y})",
            height=5,
            depth=5,
            width=1,
            caption=label,
        )
    )

    z += 2.5
    label = r"Position\\Filters" if i == 0 else ""
    legs.append(
        to_Conv(
            f"filter_{i}",
            5,
            5,
            offset=f"({z}, {x}, {y})",
            height=5,
            depth=5,
            width=1,
            caption=label,
        )
    )

    legs.append(to_connection(f"input_{i}", f"filter_{i}"))

    z += 2
    label = r"Flatten\\Argmax" if i == 0 else ""
    legs.append(
        to_Conv(
            f"argmax_{i}",
            "",
            "",
            offset=f"({z}, {x}, {y})",
            height=1,
            depth=1,
            width=1,
            caption=label,
        )
    )

    legs.append(to_connection(f"filter_{i}", f"argmax_{i}"))

plot = [
    to_head(".."),
    to_cor(),
    to_begin(),
    *legs,
]

z += 2
plot.append(
    to_Conv(
        f"footstep_options",
        4,
        "",
        offset=f"({z}, 0, 0)",
        height=1,
        depth=4,
        width=1,
        caption="Footstep\\\\Options",
    )
)

plot.extend([to_connection(f"argmax_{i}", f"footstep_options") for i in range(4)])

low_level = -offset * 3
plot.append(
    to_Conv(
        "state_input",
        43,
        "",
        offset=f"({z},{-offset*3},0)",
        height=1,
        depth=43,
        width=1,
        caption="State\\\\Input",
    ),
)

z += 3
plot.append(
    to_Conv(
        "dense1",
        64,
        "",
        offset=f"({z},{low_level/2},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
)

plot.append(to_connection("state_input", "dense1"))
plot.append(to_connection("footstep_options", "dense1"))

z += 1.5
plot.append(
    to_Conv(
        "dense2",
        64,
        "",
        offset=f"({z},{low_level/2},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
)
plot.append(to_connection("dense1", "dense2"))

z += 1.5
plot.append(
    to_Conv(
        "dense3",
        64,
        "",
        offset=f"({z},{low_level/2},0)",
        height=1,
        depth=64,
        width=1,
        caption="Dense",
    ),
)
plot.append(to_connection("dense2", "dense3"))

z += 1.5
plot.append(
    to_Conv(
        "output",
        16,
        "",
        offset=f"({z},{low_level/2},0)",
        height=1,
        depth=16,
        width=1,
        caption="Output",
    ),
)
plot.append(to_connection("dense3", "output"))

plot.append(to_end())


def main():
    namefile = str(sys.argv[0]).split(".")[0]
    to_generate(plot, namefile + ".tex")


if __name__ == "__main__":
    main()
