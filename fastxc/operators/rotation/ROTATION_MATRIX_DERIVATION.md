# ENZ to RTZ Rotation Matrix

This note documents the matrix used by the Python rotate step. It matches the
legacy `legacy/old/rotate/rotate.c` implementation.

## Component Order

Input ENZ tensor order:

```text
EE EN EZ NE NN NZ ZE ZN ZZ
```

Output RTZ tensor order:

```text
RR RT RZ TR TT TZ ZR ZT ZZ
```

FastXC stores the virtual source in SAC event fields (`evla`, `evlo`) and the
virtual receiver in SAC station fields (`stla`, `stlo`). Therefore:

- `az` is the source-to-receiver azimuth.
- `baz` is the receiver-to-source back-azimuth.

## Basis Rotation

For the virtual source side:

```text
Rs =  Es * sin(az) + Ns * cos(az)
Ts =  Es * cos(az) - Ns * sin(az)
Zs =  Zs
```

For the virtual receiver side, FastXC follows the historical convention that
receiver radial points back toward the source:

```text
Rr = -Er * sin(baz) - Nr * cos(baz)
Tr = -Er * cos(baz) + Nr * sin(baz)
Zr =  Zr
```

Each output tensor component is the product expansion of one source-side basis
term and one receiver-side basis term. For example:

```text
RR = Rs * Rr
   = (Es sin(az) + Ns cos(az)) *
     (-Er sin(baz) - Nr cos(baz))

RR = EE * -sin(az)sin(baz)
   + EN * -sin(az)cos(baz)
   + NE * -cos(az)sin(baz)
   + NN * -cos(az)cos(baz)
```

The same expansion gives the full matrix:

```text
RR [ -sinA sinB  -sinA cosB   0  -cosA sinB  -cosA cosB   0   0       0       0 ]
RT [ -sinA cosB   sinA sinB   0  -cosA cosB   cosA sinB   0   0       0       0 ]
RZ [  0           0           sinA 0           0           cosA 0       0       0 ]
TR [ -cosA sinB  -cosA cosB   0   sinA sinB   sinA cosB   0   0       0       0 ]
TT [ -cosA cosB   cosA sinB   0   sinA cosB  -sinA sinB   0   0       0       0 ]
TZ [  0           0           cosA 0           0          -sinA 0       0       0 ]
ZR [  0           0           0    0           0           0  -sinB  -cosB    0 ]
ZT [  0           0           0    0           0           0  -cosB   sinB    0 ]
ZZ [  0           0           0    0           0           0   0       0       1 ]
```

where `A = az` and `B = baz`.

## Quick Verification

Run the small deterministic check:

```bash
python fastxc/operators/rotation/verify_rotation_example.py
```

It verifies that the matrix multiplication matches explicit component-by-
component formulas for random synthetic ENZ traces.
