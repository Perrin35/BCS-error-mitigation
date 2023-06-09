OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4718[3];
rz(-2.2379212) q[0];
sx q[0];
rz(-0.18440288) q[0];
sx q[0];
rz(1.3159428) q[0];
rz(-2.7749825) q[1];
sx q[1];
rz(-0.82645196) q[1];
sx q[1];
rz(-1.1452823) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
rz(-1.8372705) q[2];
sx q[2];
rz(-0.47372813) q[2];
sx q[2];
rz(-0.0082455779) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
x q[0];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-1.8256499) q[0];
sx q[0];
rz(-2.9571898) q[0];
sx q[0];
rz(-0.90367142) q[0];
rz(-3.1333471) q[1];
sx q[1];
rz(0.47372813) q[1];
sx q[1];
rz(1.9963103) q[2];
sx q[2];
rz(-2.3151407) q[2];
sx q[2];
rz(-0.36661015) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4718[0];
measure q[1] -> c4718[1];
measure q[2] -> c4718[2];
