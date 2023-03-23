OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4515[3];
rz(4.9703496) q[0];
sx q[0];
rz(5.8196998) q[0];
sx q[0];
rz(10.296205) q[0];
rz(1.8235091) q[1];
sx q[1];
rz(-2.1967407) q[1];
sx q[1];
rz(2.9304005) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
rz(1.4460181) q[2];
sx q[2];
rz(-0.13215346) q[2];
sx q[2];
rz(0.62079566) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
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
x q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.8714269) q[0];
sx q[0];
rz(-2.6781071) q[0];
sx q[0];
rz(-1.828757) q[0];
rz(-0.62079566) q[1];
sx q[1];
rz(0.13215346) q[1];
sx q[1];
rz(-0.21119213) q[2];
sx q[2];
rz(-0.94485192) q[2];
sx q[2];
rz(1.3180836) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4515[0];
measure q[1] -> c4515[1];
measure q[2] -> c4515[2];