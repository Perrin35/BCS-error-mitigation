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
rz(-1.0415145) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.4415146) q[1];
sx q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
rz(-0.1) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-1.6000782) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.6000781) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.8707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(-pi) q[0];
rz(-0.26548755) q[1];
sx q[1];
rz(2.8761051) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-2.8761051) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.8761051) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(0.1) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.90277) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.4388227) q[1];
sx q[1];
rz(-pi) q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(0.1) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(1.40277) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.7388227) q[1];
x q[2];
rz(1.6707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.1657141) q[1];
sx q[1];
rz(-pi) q[1];
rz(-2.1657141) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
sx q[1];
rz(-0.97587855) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(2.1657141) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.8001256) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-0.058532954) q[1];
sx q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(-3.0415927) q[1];
rz(pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-2.9830597) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.15853295) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.2707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
rz(1.1586698) q[1];
sx q[1];
rz(-pi) q[1];
rz(1.1586697) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
sx q[1];
rz(1.9829229) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.982923) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(pi/2) q[1];
sx q[1];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(0.1) q[1];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.1671948) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.1743979) q[1];
sx q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-1.4743979) q[0];
sx q[1];
rz(-1.4743979) q[1];
x q[2];
rz(1.6707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.8785496) q[1];
sx q[1];
rz(-1.263043) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.2630431) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.263043) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(-0.1) q[1];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.42312167) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-0.82312165) q[1];
sx q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-0.1) q[1];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(0.92312167) q[0];
sx q[1];
rz(0.92312165) q[1];
rz(-1.8707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(-1.3749206) q[1];
sx q[1];
rz(-1.3749206) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[1];
rz(-1.7666721) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.3749206) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(3.0415927) q[1];
rz(-pi) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3154233) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.5154233) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(3.0415927) q[1];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-1.8154233) q[0];
rz(-pi) q[1];
sx q[1];
rz(-1.8154233) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.6707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(2.9588239) q[1];
sx q[1];
rz(pi) q[1];
rz(-0.18276878) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
sx q[1];
rz(0.18276875) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-2.9588239) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.2217715) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.51982115) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(2.7217715) q[0];
sx q[1];
rz(-0.41982115) q[1];
rz(-1.8707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.6780192) q[1];
sx q[1];
rz(-pi) q[1];
rz(-1.6780193) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
sx q[1];
rz(1.6780192) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.4635734) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[0];
rz(pi) q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
rz(-pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.34443166) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.1444317) q[1];
sx q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(3.0415927) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(0.84443166) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.84443165) q[1];
sx q[2];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.6254113) q[1];
sx q[1];
x q[1];
rz(1.5161814) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
sx q[1];
rz(1.6254113) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.5161814) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(3.1109629) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.7109629) q[1];
sx q[1];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-0.1) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-2.6109629) q[0];
sx q[1];
rz(-2.6109629) q[1];
rz(1.2707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
rz(-1.001327) q[1];
sx q[1];
rz(2.1402657) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-2.1402657) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.001327) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-3.0415927) q[0];
sx q[0];
x q[1];
rz(0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.4851223) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.6851223) q[1];
sx q[1];
rz(pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(-3.0415927) q[1];
rz(-pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(1.1564703) q[0];
sx q[1];
rz(-1.9851223) q[1];
rz(-1.4707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(2.8042197) q[1];
sx q[1];
rz(-pi) q[1];
rz(2.8042197) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-2.8042197) q[1];
rz(pi/2) q[2];
sx q[2];
rz(0.33737298) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(0.1) q[1];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.4547689) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.0868237) q[1];
sx q[1];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(1.1868238) q[0];
sx q[1];
rz(1.1868237) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.8707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[0];
rz(-pi/2) q[0];
rz(2.1135933) q[1];
sx q[1];
rz(2.1135933) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
sx q[1];
rz(-2.1135933) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.1135933) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/2) q[1];
sx q[1];
rz(-pi) q[1];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(3.0415927) q[1];
rz(pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.7142371) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.2273556) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
rz(pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-1.2142371) q[0];
sx q[1];
rz(-1.2142371) q[1];
rz(-pi) q[2];
sx q[2];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
rz(-1.7867062) q[1];
sx q[1];
rz(1.3548865) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
sx q[1];
rz(1.7867062) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.3548865) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
x q[0];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
sx q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.4496814) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.0919113) q[1];
sx q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
x q[1];
rz(-0.1) q[1];
rz(pi/2) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-0.94968138) q[0];
rz(-pi) q[1];
sx q[1];
rz(-0.94968135) q[1];
rz(-pi) q[2];
sx q[2];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(0.16174882) q[1];
sx q[1];
rz(-2.9798438) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
sx q[1];
rz(-0.16174882) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-0.16174885) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-0.1) q[0];
sx q[0];
rz(-0.1) q[1];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(-pi/2) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-0.8769627) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-1.6769627) q[1];
sx q[1];
rz(-pi) q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-0.1) q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(1.3769627) q[0];
rz(-pi) q[1];
sx q[1];
rz(1.3769627) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.4707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-0.92425085) q[1];
sx q[1];
rz(2.2173418) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(3.0415927) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
sx q[1];
rz(-2.2173418) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-2.2173418) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(pi) q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-0.1) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.5319858) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.2096069) q[1];
sx q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-0.1) q[0];
sx q[0];
x q[1];
rz(3.0415927) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(2.0319858) q[0];
sx q[1];
rz(2.0319858) q[1];
x q[2];
rz(-1.8707963) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(2.1402657) q[1];
sx q[1];
rz(-pi) q[1];
rz(2.1402657) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi) q[0];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
sx q[1];
rz(1.001327) q[1];
rz(pi/2) q[2];
sx q[2];
rz(1.001327) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
sx q[0];
rz(-pi/2) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi/2) q[0];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi/2) q[1];
sx q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-3.0415927) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/2) q[2];
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
