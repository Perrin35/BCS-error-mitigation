OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(-pi) q[0];
sx q[0];
rz(0.42975437) q[0];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(2.7118383) q[2];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.8987504) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.84284235) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-0.74284231) q[0];
rz(-pi) q[1];
sx q[1];
rz(-0.74284235) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4365875) q[1];
sx q[1];
x q[1];
rz(-1.7050052) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(1.7050052) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.7050052) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.8779058) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.063687) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.3779058) q[0];
sx q[1];
rz(-1.3779057) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.4437255) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.4437255) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.4437255) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.4437255) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.25014502) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.4914476) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-2.3914476) q[0];
rz(-pi) q[1];
sx q[1];
rz(-2.3914476) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.3438711) q[1];
sx q[1];
rz(-pi) q[1];
rz(-2.3438711) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(2.3438711) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-0.79772155) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.2888164) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.4888164) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(1.3527763) q[0];
sx q[1];
rz(-1.7888164) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.6147617) q[1];
sx q[1];
x q[1];
rz(-1.526831) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-1.6147617) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.6147617) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.9119584) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.3119584) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-0.72963425) q[0];
sx q[1];
rz(-0.72963422) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.27244005) q[1];
sx q[1];
rz(-pi) q[1];
rz(-0.27244006) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(0.27244005) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.8691526) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.3211188) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.1211189) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(1.8211188) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.8211189) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.4296136) q[1];
sx q[1];
rz(1.711979) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.4296136) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.4296137) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.77333475) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.1733348) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-1.8682579) q[0];
sx q[1];
rz(1.2733348) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.25959715) q[1];
sx q[1];
rz(2.8819955) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.8819955) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.25959714) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.327827) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.527827) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(1.3137657) q[0];
sx q[1];
rz(-1.827827) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.6458172) q[1];
sx q[1];
rz(1.6458172) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(1.4957755) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.6458172) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-3.1089282) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.7742571) q[1];
sx q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-2.6742571) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.46733555) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.9320394) q[1];
sx q[1];
rz(-pi) q[1];
rz(-2.9320394) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-0.20955328) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(2.9320394) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.2348775) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.7067152) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(1.4067152) q[0];
sx q[1];
rz(1.4067152) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.51468685) q[1];
sx q[1];
rz(-0.51468684) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-2.6269058) q[1];
rz(pi/2) q[2];
sx q[2];
rz(0.51468684) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.6563373) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-3.0563373) q[1];
sx q[1];
rz(pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(0.014744667) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.014744654) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.62542) q[1];
sx q[1];
x q[1];
rz(2.62542) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.62542) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.51617264) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
