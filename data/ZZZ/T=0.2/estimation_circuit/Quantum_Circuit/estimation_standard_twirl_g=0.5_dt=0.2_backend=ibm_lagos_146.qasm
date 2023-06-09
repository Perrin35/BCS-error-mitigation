OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4646[3];
rz(-1.3332869) q[0];
sx q[0];
rz(-0.3173978) q[0];
sx q[0];
rz(-2.4003098) q[0];
rz(0.080656493) q[1];
sx q[1];
rz(3.2288864) q[1];
sx q[1];
rz(11.107364) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(3.1316816) q[2];
sx q[2];
rz(3.8739825) q[2];
sx q[2];
rz(13.204063) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
x q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
x q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
rz(-pi) q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(-0.74128283) q[0];
sx q[0];
rz(-0.3173978) q[0];
sx q[0];
rz(1.3332869) q[0];
rz(2.5039007) q[1];
sx q[1];
rz(2.4092028) q[1];
sx q[1];
rz(-1.6825861) q[2];
sx q[2];
rz(-3.0542989) q[2];
sx q[2];
rz(-0.080656493) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4646[0];
measure q[1] -> c4646[1];
measure q[2] -> c4646[2];
