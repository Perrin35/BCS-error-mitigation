OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4645[3];
rz(1.2588291) q[0];
sx q[0];
rz(4.9214599) q[0];
sx q[0];
rz(11.78763) q[0];
rz(3.3842422) q[1];
sx q[1];
rz(4.6397836) q[1];
sx q[1];
rz(10.434105) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
rz(0.89410798) q[2];
sx q[2];
rz(-0.4258087) q[2];
sx q[2];
rz(-1.2793956) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
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
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-2.3628523) q[0];
sx q[0];
rz(-1.3617254) q[0];
sx q[0];
rz(-1.2588291) q[0];
rz(1.2793956) q[1];
sx q[1];
rz(-0.4258087) q[1];
sx q[1];
rz(1.0093275) q[2];
sx q[2];
rz(-1.4981909) q[2];
sx q[2];
rz(-0.24264959) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4645[0];
measure q[1] -> c4645[1];
measure q[2] -> c4645[2];
