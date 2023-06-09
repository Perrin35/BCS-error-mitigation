OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4711[3];
rz(-2.5758809) q[0];
sx q[0];
rz(-1.4737975) q[0];
sx q[0];
rz(2.9966472) q[0];
rz(1.7587288) q[1];
sx q[1];
rz(-2.1201513) q[1];
sx q[1];
rz(2.1063419) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[1];
x q[1];
rz(1.8460366) q[2];
sx q[2];
rz(4.3358039) q[2];
sx q[2];
rz(10.017688) q[2];
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
x q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(2.9966472) q[0];
sx q[0];
rz(-1.6677952) q[0];
sx q[0];
rz(-0.56571172) q[0];
rz(-2.5486829) q[1];
sx q[1];
rz(-1.1942112) q[1];
sx q[1];
rz(1.0352507) q[2];
sx q[2];
rz(-2.1201513) q[2];
sx q[2];
rz(-1.7587288) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4711[0];
measure q[1] -> c4711[1];
measure q[2] -> c4711[2];
