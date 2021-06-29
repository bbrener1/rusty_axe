#!/bin/bash

# rustyaxe is a shortcut to the workbook directory
echo "#!/bin/bash" > $PREFIX/rustyaxe
echo "cd $PREFIX/work" >> $PREFIX/rustyaxe
chmod +x $PREFIX/rustyaxe
