OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c[3];
rz(1.36033745335607) q[0];
sx q[0];
rz(4.84625486784630) q[0];
sx q[0];
rz(14.1381188908496) q[0];
rz(3.04344358660845) q[1];
sx q[1];
rz(5.57690824791216) q[1];
sx q[1];
rz(14.2248885167272) q[1];
rz(4.79136697766011) q[2];
sx q[2];
rz(3.59445947526754) q[2];
sx q[2];
rz(12.0834843199813) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(-pi/2) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[0];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
rz(pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
x q[1];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
rz(-pi) q[2];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[0];
x q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
x q[1];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
x q[0];
sx q[2];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[1];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(-pi) q[1];
x q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(pi) q[1];
rz(pi/2) q[2];
sx q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
rz(-pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
x q[1];
x q[2];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[2];
sx q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
x q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi/2) q[2];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
sx q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
rz(pi) q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2];
x q[0];
rz(pi/2) q[0];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
x q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[0],q[1],q[2];
cx q[2],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
sx q[0];
rz(-pi) q[1];
x q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi/2) q[0];
sx q[0];
rz(-pi/2) q[0];
barrier q[0],q[1],q[2];
cx q[1],q[2];
barrier q[0],q[1],q[2];
rz(-pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[0];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2];
rz(pi) q[1];
rz(-pi/2) q[2];
barrier q[0],q[1],q[2];
cx q[0],q[1];
barrier q[0],q[1],q[2];
rz(-pi) q[0];
rz(-pi) q[1];
rz(pi/2) q[2];
barrier q[0],q[1],q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
rz(-4.71334093008021) q[0];
sx q[0];
rz(1.43693043933329) q[0];
sx q[0];
rz(8.06444050741331) q[0];
rz(-2.65870635921194) q[1];
sx q[1];
rz(2.68872583191205) q[1];
sx q[1];
rz(4.63341098310927) q[1];
rz(-4.80011055595787) q[2];
sx q[2];
rz(0.706277059267426) q[2];
sx q[2];
rz(6.38133437416093) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
