while pgrep -f intentonomy_vit_mcc.sh > /dev/null; do
    sleep 1000
done

bash scripts/intentonomy_vit_mcc.sh
