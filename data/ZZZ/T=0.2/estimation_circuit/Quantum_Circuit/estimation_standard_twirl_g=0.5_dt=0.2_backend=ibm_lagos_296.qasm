OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4796[3];
rz(0.10535333) q[0];
sx q[0];
rz(-1.656266) q[0];
sx q[0];
rz(-0.74946122) q[0];
rz(-1.9460497) q[1];
sx q[1];
rz(-1.0927967) q[1];
sx q[1];
rz(1.4105295) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-pi) q[1];
rz(2.0545788) q[2];
sx q[2];
rz(-3.0096141) q[2];
sx q[2];
rz(1.626057) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
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
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.3921314) q[0];
sx q[0];
rz(-1.4853266) q[0];
sx q[0];
rz(3.0362393) q[0];
rz(-1.5155356) q[1];
sx q[1];
rz(0.13197858) q[1];
sx q[1];
rz(1.4105295) q[2];
sx q[2];
rz(-2.0487959) q[2];
sx q[2];
rz(-1.195543) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4796[0];
measure q[1] -> c4796[1];
measure q[2] -> c4796[2];
