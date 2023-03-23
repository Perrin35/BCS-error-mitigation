OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4716[3];
rz(3.1312878) q[0];
sx q[0];
rz(-1.8281653) q[0];
sx q[0];
rz(-1.1489073) q[0];
rz(1.8538148) q[1];
sx q[1];
rz(3.9625075) q[1];
sx q[1];
rz(10.511973) q[1];
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
x q[0];
x q[1];
rz(-0.42640029) q[2];
sx q[2];
rz(-0.90123547) q[2];
sx q[2];
rz(-1.7636926) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-pi) q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(1.1489073) q[0];
sx q[0];
rz(-1.8281653) q[0];
sx q[0];
rz(-3.1312878) q[0];
rz(1.7636926) q[1];
sx q[1];
rz(-0.90123547) q[1];
sx q[1];
rz(1.087195) q[2];
sx q[2];
rz(-0.82091487) q[2];
sx q[2];
rz(1.2877779) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4716[0];
measure q[1] -> c4716[1];
measure q[2] -> c4716[2];