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
rz(-2.4528156) q[0];
sx q[0];
rz(pi/2) q[0];
rz(0.28877705) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(2.9528156) q[0];
sx q[1];
rz(2.9528156) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.979367) q[1];
sx q[1];
rz(1.9793669) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
sx q[1];
rz(-1.979367) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.9793669) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
rz(-pi) q[0];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.7020106) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(1.9020106) q[1];
sx q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
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
rz(0.93958203) q[0];
sx q[1];
rz(-2.2020106) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.30017865) q[1];
sx q[1];
rz(-2.841414) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
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
rz(2.841414) q[1];
rz(pi/2) q[2];
sx q[2];
rz(2.841414) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
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
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.4528156) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.8528156) q[1];
sx q[1];
rz(pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(0.1) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi/2) q[0];
sx q[0];
rz(-0.18877709) q[0];
rz(-pi) q[1];
sx q[1];
rz(2.9528156) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.577748) q[1];
sx q[1];
rz(-pi) q[1];
rz(1.5638446) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-0.1) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.577748) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(1.5777481) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
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
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-2.3594611) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-0.017868354) q[1];
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
rz(-0.2821316) q[0];
rz(-pi) q[1];
sx q[1];
rz(-0.28213155) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3369261) q[1];
sx q[1];
x q[1];
rz(-0.80466656) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-0.1) q[1];
sx q[2];
rz(-0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(0.80466655) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(0.80466656) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
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
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(pi/2) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(0.1) q[0];
sx q[0];
rz(-pi) q[0];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
sx q[1];
rz(-pi/2) q[1];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(0.20440142) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-0.19559855) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
sx q[0];
rz(3.0415927) q[0];
sx q[0];
x q[1];
rz(-3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(-2.8459941) q[0];
rz(-pi) q[1];
sx q[1];
rz(0.29559855) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.979367) q[1];
sx q[1];
rz(1.9793669) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(3.0415927) q[1];
sx q[2];
rz(0.1) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-1.979367) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-1.9793669) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
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
rz(3.0415927) q[0];
sx q[0];
rz(-pi) q[0];
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-1.0814073) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-1.8814073) q[1];
sx q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
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
rz(-1.5601854) q[0];
sx q[1];
rz(1.5814073) q[1];
rz(0.1) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.7666721) q[1];
sx q[1];
rz(pi) q[1];
rz(-1.3749206) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(-3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(1.3749206) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(-1.7666721) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
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
x q[1];
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
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(1.5200022) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-2.0215905) q[1];
sx q[1];
x q[1];
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
rz(pi/2) q[0];
sx q[0];
rz(2.1215905) q[0];
sx q[1];
rz(2.1215905) q[1];
rz(-0.3) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(2.3568181) q[1];
sx q[1];
rz(2.3568181) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-3.0415927) q[1];
sx q[2];
rz(3.0415927) q[2];
sx q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
sx q[1];
rz(-2.3568181) q[1];
rz(pi/2) q[2];
sx q[2];
rz(0.78477458) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
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
x q[1];
rz(3.0415927) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
sx q[1];
rz(pi/2) q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
