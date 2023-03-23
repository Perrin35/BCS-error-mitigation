OPENQASM 2.0;
include "qelib1.inc";
qreg q[7];
creg c4568[3];
rz(2.670432) q[0];
sx q[0];
rz(-2.7306705) q[0];
sx q[0];
rz(0.66517107) q[0];
rz(5.1664275) q[1];
sx q[1];
rz(5.1369527) q[1];
sx q[1];
rz(14.5) q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[0];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.70038425) q[2];
sx q[2];
rz(4.0761192) q[2];
sx q[2];
rz(12.948156) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
rz(-pi) q[1];
x q[1];
rz(-pi) q[2];
barrier q[2],q[1];
cx q[2],q[1];
barrier q[2],q[1];
rz(-pi) q[1];
rz(-pi) q[2];
barrier q[1],q[2];
cx q[1],q[2];
barrier q[1],q[2];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
x q[1];
barrier q[0],q[1];
cx q[0],q[1];
barrier q[0],q[1];
rz(0.66517107) q[0];
sx q[0];
rz(-0.4109222) q[0];
sx q[0];
rz(0.47116061) q[0];
rz(2.759807) q[1];
sx q[1];
rz(2.2070661) q[1];
sx q[1];
rz(1.2079633) q[2];
sx q[2];
rz(-1.1462326) q[2];
sx q[2];
rz(1.1167578) q[2];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6];
measure q[0] -> c4568[0];
measure q[1] -> c4568[1];
measure q[2] -> c4568[2];
